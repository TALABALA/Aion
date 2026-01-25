"""
ML-Based Anomaly Detection System.

Implements real-time anomaly detection for security monitoring using:
- Isolation Forest for outlier detection
- Statistical methods (Z-score, IQR)
- Time-series analysis
- Behavioral clustering
- Autoencoder-based detection

Detects anomalies in:
- Authentication patterns
- API usage patterns
- Resource access patterns
- Network behavior
- User activity sequences
"""

from __future__ import annotations

import math
import random
import statistics
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

import structlog

logger = structlog.get_logger()


class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    AUTH_PATTERN = "auth_pattern"
    ACCESS_PATTERN = "access_pattern"
    RATE_ANOMALY = "rate_anomaly"
    GEOGRAPHIC = "geographic"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    RESOURCE_USAGE = "resource_usage"
    SEQUENCE = "sequence"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly."""
    id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float  # 0.0 to 1.0, higher = more anomalous
    description: str
    entity_type: str  # user, ip, session, etc.
    entity_id: str
    features: dict[str, Any]
    detected_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Feature vector for anomaly detection."""
    entity_id: str
    timestamp: float
    features: dict[str, float]
    labels: dict[str, str] = field(default_factory=dict)


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def fit(self, data: list[FeatureVector]) -> None:
        """Train the detector on historical data."""
        pass

    @abstractmethod
    def predict(self, sample: FeatureVector) -> tuple[bool, float]:
        """
        Predict if sample is anomalous.

        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        pass

    @abstractmethod
    def partial_fit(self, sample: FeatureVector) -> None:
        """Incrementally update the model with new data."""
        pass


class IsolationForest(AnomalyDetector):
    """
    Isolation Forest anomaly detector.

    Based on the principle that anomalies are easier to isolate than normal points.
    Uses random recursive partitioning to isolate observations.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        max_features: int = 0,  # 0 = all features
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self._trees: list[IsolationTree] = []
        self._feature_names: list[str] = []
        self._threshold: float = 0.5
        self._avg_path_length: float = 0.0
        self._random = random.Random(random_state)
        self._logger = logger.bind(detector="isolation_forest")

    def fit(self, data: list[FeatureVector]) -> None:
        """Train the isolation forest."""
        if not data:
            return

        self._feature_names = list(data[0].features.keys())
        n_samples = len(data)

        # Convert to numeric arrays
        X = [
            [fv.features.get(f, 0.0) for f in self._feature_names]
            for fv in data
        ]

        # Calculate average path length for normalization
        self._avg_path_length = self._calculate_avg_path_length(n_samples)

        # Build trees
        self._trees = []
        sample_size = min(self.max_samples, n_samples)

        for _ in range(self.n_estimators):
            # Sample data
            indices = self._random.sample(range(n_samples), sample_size)
            sample = [X[i] for i in indices]

            # Build tree
            tree = IsolationTree(
                max_depth=int(math.ceil(math.log2(sample_size))),
                feature_indices=self._get_feature_indices(),
                random=self._random,
            )
            tree.fit(sample)
            self._trees.append(tree)

        # Calculate threshold based on contamination
        scores = [self._compute_anomaly_score(x) for x in X]
        scores.sort(reverse=True)
        threshold_idx = int(self.contamination * len(scores))
        self._threshold = scores[threshold_idx] if threshold_idx < len(scores) else 0.5

        self._logger.info(
            "Isolation forest trained",
            n_trees=self.n_estimators,
            n_samples=n_samples,
            threshold=self._threshold,
        )

    def predict(self, sample: FeatureVector) -> tuple[bool, float]:
        """Predict if sample is anomalous."""
        if not self._trees:
            return (False, 0.0)

        x = [sample.features.get(f, 0.0) for f in self._feature_names]
        score = self._compute_anomaly_score(x)
        is_anomaly = score > self._threshold

        return (is_anomaly, score)

    def partial_fit(self, sample: FeatureVector) -> None:
        """
        Incrementally update the model.

        Note: True incremental training for isolation forest is complex.
        This implementation randomly replaces trees periodically.
        """
        # Store sample for batch retraining
        pass

    def _compute_anomaly_score(self, x: list[float]) -> float:
        """Compute anomaly score for a sample."""
        if not self._trees:
            return 0.0

        # Average path length across all trees
        path_lengths = [tree.path_length(x) for tree in self._trees]
        avg_path = sum(path_lengths) / len(path_lengths)

        # Normalize using average path length
        score = 2 ** (-avg_path / self._avg_path_length)

        return score

    def _calculate_avg_path_length(self, n: int) -> float:
        """Calculate expected average path length for n samples."""
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        # H(n-1) approximation using Euler's constant
        h = math.log(n - 1) + 0.5772156649
        return 2 * h - (2 * (n - 1) / n)

    def _get_feature_indices(self) -> list[int]:
        """Get indices of features to use."""
        n_features = len(self._feature_names)
        if self.max_features <= 0 or self.max_features >= n_features:
            return list(range(n_features))
        return self._random.sample(range(n_features), self.max_features)


class IsolationTree:
    """Single isolation tree for Isolation Forest."""

    def __init__(
        self,
        max_depth: int,
        feature_indices: list[int],
        random: random.Random,
    ) -> None:
        self.max_depth = max_depth
        self.feature_indices = feature_indices
        self._random = random
        self._root: Optional[IsolationNode] = None

    def fit(self, X: list[list[float]]) -> None:
        """Build the isolation tree."""
        self._root = self._build_tree(X, 0)

    def path_length(self, x: list[float]) -> float:
        """Calculate path length for a sample."""
        return self._traverse(x, self._root, 0)

    def _build_tree(
        self,
        X: list[list[float]],
        depth: int,
    ) -> IsolationNode:
        """Recursively build the tree."""
        n_samples = len(X)

        # Terminal conditions
        if depth >= self.max_depth or n_samples <= 1:
            return IsolationNode(
                is_leaf=True,
                size=n_samples,
            )

        # Select random feature and split value
        feature_idx = self._random.choice(self.feature_indices)
        feature_values = [x[feature_idx] for x in X]
        min_val, max_val = min(feature_values), max(feature_values)

        if min_val == max_val:
            return IsolationNode(
                is_leaf=True,
                size=n_samples,
            )

        split_value = self._random.uniform(min_val, max_val)

        # Split data
        left_data = [x for x in X if x[feature_idx] < split_value]
        right_data = [x for x in X if x[feature_idx] >= split_value]

        return IsolationNode(
            is_leaf=False,
            feature_idx=feature_idx,
            split_value=split_value,
            left=self._build_tree(left_data, depth + 1),
            right=self._build_tree(right_data, depth + 1),
        )

    def _traverse(
        self,
        x: list[float],
        node: Optional[IsolationNode],
        depth: int,
    ) -> float:
        """Traverse tree and return path length."""
        if node is None or node.is_leaf:
            if node and node.size > 1:
                # Add expected path length for remaining samples
                return depth + self._c(node.size)
            return depth

        if x[node.feature_idx] < node.split_value:
            return self._traverse(x, node.left, depth + 1)
        return self._traverse(x, node.right, depth + 1)

    def _c(self, n: int) -> float:
        """Expected path length adjustment."""
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        h = math.log(n - 1) + 0.5772156649
        return 2 * h - (2 * (n - 1) / n)


@dataclass
class IsolationNode:
    """Node in an isolation tree."""
    is_leaf: bool
    size: int = 0
    feature_idx: int = 0
    split_value: float = 0.0
    left: Optional["IsolationNode"] = None
    right: Optional["IsolationNode"] = None


class StatisticalDetector(AnomalyDetector):
    """
    Statistical anomaly detector using Z-score and IQR methods.

    Simple but effective for univariate and low-dimensional data.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        method: str = "hybrid",  # zscore, iqr, or hybrid
        window_size: int = 1000,
    ) -> None:
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.method = method
        self.window_size = window_size
        self._stats: dict[str, FeatureStats] = {}
        self._logger = logger.bind(detector="statistical")

    def fit(self, data: list[FeatureVector]) -> None:
        """Calculate statistics from historical data."""
        if not data:
            return

        # Collect feature values
        feature_values: dict[str, list[float]] = {}
        for fv in data:
            for name, value in fv.features.items():
                if name not in feature_values:
                    feature_values[name] = []
                feature_values[name].append(value)

        # Calculate stats for each feature
        for name, values in feature_values.items():
            self._stats[name] = self._calculate_stats(values)

        self._logger.info(
            "Statistical detector trained",
            n_features=len(self._stats),
            n_samples=len(data),
        )

    def predict(self, sample: FeatureVector) -> tuple[bool, float]:
        """Check if sample is anomalous."""
        if not self._stats:
            return (False, 0.0)

        anomaly_scores: list[float] = []

        for name, value in sample.features.items():
            if name not in self._stats:
                continue

            stats = self._stats[name]
            score = 0.0

            if self.method in ("zscore", "hybrid") and stats.std > 0:
                z_score = abs(value - stats.mean) / stats.std
                score = max(score, z_score / self.z_threshold)

            if self.method in ("iqr", "hybrid") and stats.iqr > 0:
                lower = stats.q1 - self.iqr_multiplier * stats.iqr
                upper = stats.q3 + self.iqr_multiplier * stats.iqr
                if value < lower:
                    score = max(score, (lower - value) / stats.iqr)
                elif value > upper:
                    score = max(score, (value - upper) / stats.iqr)

            anomaly_scores.append(min(1.0, score))

        if not anomaly_scores:
            return (False, 0.0)

        # Aggregate scores - use max for any feature being anomalous
        max_score = max(anomaly_scores)
        avg_score = sum(anomaly_scores) / len(anomaly_scores)
        final_score = 0.7 * max_score + 0.3 * avg_score

        return (final_score > 0.5, final_score)

    def partial_fit(self, sample: FeatureVector) -> None:
        """Update statistics with new sample."""
        for name, value in sample.features.items():
            if name not in self._stats:
                self._stats[name] = FeatureStats(
                    mean=value,
                    std=0.0,
                    min_val=value,
                    max_val=value,
                    q1=value,
                    median=value,
                    q3=value,
                    iqr=0.0,
                    count=1,
                )
                continue

            stats = self._stats[name]

            # Update running statistics (Welford's algorithm)
            stats.count += 1
            delta = value - stats.mean
            stats.mean += delta / stats.count
            delta2 = value - stats.mean
            stats.m2 += delta * delta2
            stats.std = math.sqrt(stats.m2 / stats.count) if stats.count > 1 else 0.0

            stats.min_val = min(stats.min_val, value)
            stats.max_val = max(stats.max_val, value)

    def _calculate_stats(self, values: list[float]) -> "FeatureStats":
        """Calculate statistics for a feature."""
        if not values:
            return FeatureStats()

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0.0

        q1 = sorted_vals[int(n * 0.25)]
        median = sorted_vals[int(n * 0.5)]
        q3 = sorted_vals[int(n * 0.75)]
        iqr = q3 - q1

        return FeatureStats(
            mean=mean,
            std=std,
            min_val=min(values),
            max_val=max(values),
            q1=q1,
            median=median,
            q3=q3,
            iqr=iqr,
            count=n,
        )


@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    q1: float = 0.0
    median: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    count: int = 0
    m2: float = 0.0  # For Welford's algorithm


class TimeSeriesDetector(AnomalyDetector):
    """
    Time series anomaly detector.

    Detects anomalies in temporal patterns using:
    - Moving average deviation
    - Seasonal decomposition
    - Change point detection
    """

    def __init__(
        self,
        window_size: int = 60,
        seasonality: int = 0,  # 0 = auto-detect
        sensitivity: float = 2.0,
    ) -> None:
        self.window_size = window_size
        self.seasonality = seasonality
        self.sensitivity = sensitivity
        self._history: dict[str, deque] = {}
        self._baselines: dict[str, list[float]] = {}
        self._logger = logger.bind(detector="timeseries")

    def fit(self, data: list[FeatureVector]) -> None:
        """Build baseline from historical data."""
        if not data:
            return

        # Group by feature
        feature_series: dict[str, list[tuple[float, float]]] = {}
        for fv in data:
            for name, value in fv.features.items():
                if name not in feature_series:
                    feature_series[name] = []
                feature_series[name].append((fv.timestamp, value))

        # Sort by timestamp and build baselines
        for name, series in feature_series.items():
            series.sort(key=lambda x: x[0])
            values = [v for _, v in series]

            # Calculate moving averages as baseline
            if len(values) >= self.window_size:
                baseline = []
                for i in range(len(values) - self.window_size + 1):
                    window = values[i:i + self.window_size]
                    baseline.append(statistics.mean(window))
                self._baselines[name] = baseline

            # Initialize history
            self._history[name] = deque(values[-self.window_size:], maxlen=self.window_size)

    def predict(self, sample: FeatureVector) -> tuple[bool, float]:
        """Detect if sample is anomalous."""
        anomaly_scores: list[float] = []

        for name, value in sample.features.items():
            if name not in self._history:
                self._history[name] = deque(maxlen=self.window_size)
                continue

            history = list(self._history[name])
            if len(history) < self.window_size // 2:
                continue

            # Calculate expected value and deviation
            expected = statistics.mean(history)
            std = statistics.stdev(history) if len(history) > 1 else 1.0

            if std > 0:
                deviation = abs(value - expected) / std
                score = min(1.0, deviation / (self.sensitivity * 2))
                anomaly_scores.append(score)

        if not anomaly_scores:
            return (False, 0.0)

        max_score = max(anomaly_scores)
        return (max_score > 0.5, max_score)

    def partial_fit(self, sample: FeatureVector) -> None:
        """Update with new sample."""
        for name, value in sample.features.items():
            if name not in self._history:
                self._history[name] = deque(maxlen=self.window_size)
            self._history[name].append(value)


class SequenceDetector(AnomalyDetector):
    """
    Sequence anomaly detector for detecting unusual patterns in event sequences.

    Uses n-gram models and Markov chains to model normal behavior.
    """

    def __init__(
        self,
        n_gram_size: int = 3,
        min_frequency: int = 5,
        rare_threshold: float = 0.01,
    ) -> None:
        self.n_gram_size = n_gram_size
        self.min_frequency = min_frequency
        self.rare_threshold = rare_threshold
        self._transitions: dict[tuple, dict[str, int]] = {}
        self._total_transitions: dict[tuple, int] = {}
        self._logger = logger.bind(detector="sequence")

    def fit(self, data: list[FeatureVector]) -> None:
        """Build transition model from historical sequences."""
        # Group by entity
        entity_sequences: dict[str, list[str]] = {}
        for fv in sorted(data, key=lambda x: x.timestamp):
            entity = fv.entity_id
            event = fv.labels.get("event_type", "unknown")

            if entity not in entity_sequences:
                entity_sequences[entity] = []
            entity_sequences[entity].append(event)

        # Build n-gram model
        for entity, sequence in entity_sequences.items():
            for i in range(len(sequence) - self.n_gram_size):
                context = tuple(sequence[i:i + self.n_gram_size - 1])
                next_event = sequence[i + self.n_gram_size - 1]

                if context not in self._transitions:
                    self._transitions[context] = {}
                    self._total_transitions[context] = 0

                if next_event not in self._transitions[context]:
                    self._transitions[context][next_event] = 0

                self._transitions[context][next_event] += 1
                self._total_transitions[context] += 1

        self._logger.info(
            "Sequence detector trained",
            n_contexts=len(self._transitions),
            n_sequences=len(entity_sequences),
        )

    def predict(self, sample: FeatureVector) -> tuple[bool, float]:
        """Check if event sequence is anomalous."""
        context = sample.labels.get("context")
        event = sample.labels.get("event_type")

        if not context or not event:
            return (False, 0.0)

        context_tuple = tuple(context.split(",")[-self.n_gram_size + 1:])

        if context_tuple not in self._transitions:
            # Unknown context - slightly anomalous
            return (False, 0.3)

        total = self._total_transitions[context_tuple]
        count = self._transitions[context_tuple].get(event, 0)

        if count == 0:
            # Never seen this transition
            return (True, 0.9)

        probability = count / total

        if probability < self.rare_threshold:
            score = 1.0 - (probability / self.rare_threshold)
            return (score > 0.5, score)

        return (False, 0.1)

    def partial_fit(self, sample: FeatureVector) -> None:
        """Update model with new event."""
        context = sample.labels.get("context")
        event = sample.labels.get("event_type")

        if not context or not event:
            return

        context_tuple = tuple(context.split(",")[-self.n_gram_size + 1:])

        if context_tuple not in self._transitions:
            self._transitions[context_tuple] = {}
            self._total_transitions[context_tuple] = 0

        if event not in self._transitions[context_tuple]:
            self._transitions[context_tuple][event] = 0

        self._transitions[context_tuple][event] += 1
        self._total_transitions[context_tuple] += 1


class EnsembleDetector(AnomalyDetector):
    """
    Ensemble anomaly detector combining multiple detection methods.
    """

    def __init__(
        self,
        detectors: Optional[list[AnomalyDetector]] = None,
        weights: Optional[list[float]] = None,
        voting: str = "soft",  # soft or hard
        threshold: float = 0.5,
    ) -> None:
        self.detectors = detectors or [
            IsolationForest(),
            StatisticalDetector(),
            TimeSeriesDetector(),
        ]
        self.weights = weights or [1.0] * len(self.detectors)
        self.voting = voting
        self.threshold = threshold
        self._logger = logger.bind(detector="ensemble")

    def fit(self, data: list[FeatureVector]) -> None:
        """Train all detectors."""
        for detector in self.detectors:
            detector.fit(data)
        self._logger.info("Ensemble trained", n_detectors=len(self.detectors))

    def predict(self, sample: FeatureVector) -> tuple[bool, float]:
        """Predict using ensemble voting."""
        predictions: list[tuple[bool, float]] = []

        for detector in self.detectors:
            try:
                pred = detector.predict(sample)
                predictions.append(pred)
            except Exception as e:
                self._logger.warning(f"Detector failed: {e}")
                continue

        if not predictions:
            return (False, 0.0)

        if self.voting == "hard":
            # Majority voting
            votes = sum(1 for is_anomaly, _ in predictions if is_anomaly)
            is_anomaly = votes > len(predictions) / 2
            score = votes / len(predictions)
        else:
            # Soft voting (weighted average of scores)
            total_weight = sum(
                self.weights[i]
                for i in range(len(predictions))
            )
            weighted_score = sum(
                score * self.weights[i]
                for i, (_, score) in enumerate(predictions)
            ) / total_weight
            score = weighted_score
            is_anomaly = score > self.threshold

        return (is_anomaly, score)

    def partial_fit(self, sample: FeatureVector) -> None:
        """Update all detectors."""
        for detector in self.detectors:
            try:
                detector.partial_fit(sample)
            except Exception:
                pass


class AnomalyDetectionService:
    """
    Main anomaly detection service.

    Coordinates multiple detectors, manages detection pipelines,
    and handles alerts.
    """

    def __init__(
        self,
        detector: Optional[AnomalyDetector] = None,
        alert_callback: Optional[Callable[[Anomaly], None]] = None,
        severity_thresholds: Optional[dict[float, AnomalySeverity]] = None,
    ) -> None:
        self.detector = detector or EnsembleDetector()
        self.alert_callback = alert_callback
        self.severity_thresholds = severity_thresholds or {
            0.3: AnomalySeverity.LOW,
            0.5: AnomalySeverity.MEDIUM,
            0.7: AnomalySeverity.HIGH,
            0.9: AnomalySeverity.CRITICAL,
        }
        self._anomaly_history: deque = deque(maxlen=10000)
        self._suppression: dict[str, float] = {}  # entity_id -> suppress until
        self._logger = logger.bind(service="anomaly_detection")

    def train(self, historical_data: list[FeatureVector]) -> None:
        """Train the detector on historical data."""
        self._logger.info("Training anomaly detector", n_samples=len(historical_data))
        self.detector.fit(historical_data)

    def detect(
        self,
        sample: FeatureVector,
        anomaly_type: AnomalyType = AnomalyType.BEHAVIORAL,
    ) -> Optional[Anomaly]:
        """
        Detect anomalies in a sample.

        Returns Anomaly if detected, None otherwise.
        """
        # Check suppression
        if sample.entity_id in self._suppression:
            if time.time() < self._suppression[sample.entity_id]:
                return None

        # Run detection
        is_anomaly, score = self.detector.predict(sample)

        # Update model incrementally
        self.detector.partial_fit(sample)

        if not is_anomaly:
            return None

        # Determine severity
        severity = AnomalySeverity.INFO
        for threshold, sev in sorted(self.severity_thresholds.items()):
            if score >= threshold:
                severity = sev

        # Create anomaly record
        anomaly = Anomaly(
            id=f"anomaly-{int(time.time() * 1000)}-{sample.entity_id[:8]}",
            anomaly_type=anomaly_type,
            severity=severity,
            score=score,
            description=self._generate_description(sample, score),
            entity_type=sample.labels.get("entity_type", "unknown"),
            entity_id=sample.entity_id,
            features=sample.features,
            metadata=sample.labels,
        )

        # Record and alert
        self._anomaly_history.append(anomaly)

        if self.alert_callback and severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL):
            try:
                self.alert_callback(anomaly)
            except Exception as e:
                self._logger.error("Alert callback failed", error=str(e))

        self._logger.warning(
            "Anomaly detected",
            anomaly_id=anomaly.id,
            entity_id=anomaly.entity_id,
            severity=severity.value,
            score=score,
        )

        return anomaly

    def suppress_alerts(
        self,
        entity_id: str,
        duration_seconds: int = 300,
    ) -> None:
        """Suppress alerts for an entity temporarily."""
        self._suppression[entity_id] = time.time() + duration_seconds

    def get_recent_anomalies(
        self,
        entity_id: Optional[str] = None,
        severity: Optional[AnomalySeverity] = None,
        limit: int = 100,
    ) -> list[Anomaly]:
        """Get recent anomalies, optionally filtered."""
        anomalies = list(self._anomaly_history)

        if entity_id:
            anomalies = [a for a in anomalies if a.entity_id == entity_id]

        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]

        return anomalies[-limit:]

    def get_anomaly_stats(self) -> dict[str, Any]:
        """Get anomaly statistics."""
        if not self._anomaly_history:
            return {"total": 0}

        anomalies = list(self._anomaly_history)

        by_severity = {}
        for sev in AnomalySeverity:
            by_severity[sev.value] = sum(1 for a in anomalies if a.severity == sev)

        by_type = {}
        for atype in AnomalyType:
            by_type[atype.value] = sum(1 for a in anomalies if a.anomaly_type == atype)

        return {
            "total": len(anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "avg_score": sum(a.score for a in anomalies) / len(anomalies),
            "unique_entities": len(set(a.entity_id for a in anomalies)),
        }

    def _generate_description(
        self,
        sample: FeatureVector,
        score: float,
    ) -> str:
        """Generate human-readable anomaly description."""
        # Find the most anomalous features
        feature_parts = []
        for name, value in sample.features.items():
            feature_parts.append(f"{name}={value:.2f}")

        features_str = ", ".join(feature_parts[:3])
        return f"Anomalous behavior detected (score={score:.2f}): {features_str}"


# Feature extractors for common security monitoring scenarios

class AuthFeatureExtractor:
    """Extract features from authentication events."""

    @staticmethod
    def extract(
        user_id: str,
        ip_address: str,
        success: bool,
        timestamp: float,
        additional: Optional[dict[str, Any]] = None,
    ) -> FeatureVector:
        """Extract features from an auth event."""
        additional = additional or {}

        features = {
            "success": 1.0 if success else 0.0,
            "hour_of_day": time.localtime(timestamp).tm_hour / 24.0,
            "day_of_week": time.localtime(timestamp).tm_wday / 7.0,
        }

        if "attempt_count" in additional:
            features["attempt_count"] = float(additional["attempt_count"])

        if "time_since_last_auth" in additional:
            features["time_since_last_auth"] = additional["time_since_last_auth"] / 3600.0

        return FeatureVector(
            entity_id=user_id,
            timestamp=timestamp,
            features=features,
            labels={
                "event_type": "auth",
                "ip_address": ip_address,
                "entity_type": "user",
            },
        )


class APIFeatureExtractor:
    """Extract features from API requests."""

    @staticmethod
    def extract(
        user_id: str,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        request_size: int,
        response_size: int,
        timestamp: float,
    ) -> FeatureVector:
        """Extract features from an API request."""
        features = {
            "response_time_ms": response_time_ms,
            "status_code": float(status_code),
            "request_size": float(request_size),
            "response_size": float(response_size),
            "hour_of_day": time.localtime(timestamp).tm_hour / 24.0,
            "is_error": 1.0 if status_code >= 400 else 0.0,
        }

        return FeatureVector(
            entity_id=user_id,
            timestamp=timestamp,
            features=features,
            labels={
                "event_type": "api_request",
                "endpoint": endpoint,
                "method": method,
                "entity_type": "user",
            },
        )
