"""
AION Anomaly Detector

Detect anomalies in metrics using:
- Statistical methods (z-score, IQR)
- Moving averages
- Seasonal decomposition
- Machine learning (isolation forest)
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from aion.observability.types import Anomaly, AnomalyType, AlertSeverity
from aion.observability.metrics.engine import MetricsEngine

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection."""
    # Z-score threshold for anomaly detection
    z_score_threshold: float = 3.0

    # Minimum samples needed for detection
    min_samples: int = 30

    # Moving average window
    moving_average_window: int = 10

    # Seasonal period (e.g., 24 for hourly data with daily pattern)
    seasonal_period: int = 24

    # Sensitivity (lower = more sensitive)
    sensitivity: float = 1.0


@dataclass
class MetricState:
    """State for tracking a single metric."""
    name: str
    labels: Dict[str, str] = field(default_factory=dict)

    # Historical values
    values: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')

    # Moving average
    moving_avg: float = 0.0
    moving_std: float = 0.0

    # Trend
    trend: float = 0.0  # Positive = increasing, negative = decreasing

    # Last anomaly
    last_anomaly_time: Optional[datetime] = None
    consecutive_anomalies: int = 0


class AnomalyDetector:
    """
    SOTA Anomaly detection for metrics.

    Features:
    - Multiple detection algorithms
    - Adaptive thresholds
    - Seasonal pattern detection
    - Anomaly correlation
    """

    def __init__(
        self,
        metrics_engine: MetricsEngine,
        config: AnomalyDetectorConfig = None,
        check_interval: float = 60.0,
    ):
        self.metrics = metrics_engine
        self.config = config or AnomalyDetectorConfig()
        self.check_interval = check_interval

        # Tracked metrics
        self._metric_states: Dict[str, MetricState] = {}

        # Detected anomalies
        self._anomalies: List[Anomaly] = []
        self._max_anomalies = 10000

        # Callbacks
        self._anomaly_callbacks: List[Callable[[Anomaly], None]] = []

        # Metrics to monitor
        self._monitored_metrics: Dict[str, Dict[str, Any]] = {}

        # Background task
        self._detection_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the anomaly detector."""
        if self._initialized:
            return

        logger.info("Initializing Anomaly Detector")

        # Start detection loop
        self._detection_task = asyncio.create_task(self._detection_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the anomaly detector."""
        self._shutdown_event.set()

        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    def add_anomaly_callback(self, callback: Callable[[Anomaly], None]) -> None:
        """Add a callback for detected anomalies."""
        self._anomaly_callbacks.append(callback)

    def monitor_metric(
        self,
        name: str,
        labels: Dict[str, str] = None,
        z_score_threshold: float = None,
        min_samples: int = None,
    ) -> None:
        """Add a metric to monitor for anomalies."""
        key = self._metric_key(name, labels or {})
        self._monitored_metrics[key] = {
            "name": name,
            "labels": labels or {},
            "z_score_threshold": z_score_threshold or self.config.z_score_threshold,
            "min_samples": min_samples or self.config.min_samples,
        }

        # Initialize state
        if key not in self._metric_states:
            self._metric_states[key] = MetricState(name=name, labels=labels or {})

    def stop_monitoring(self, name: str, labels: Dict[str, str] = None) -> bool:
        """Stop monitoring a metric."""
        key = self._metric_key(name, labels or {})
        if key in self._monitored_metrics:
            del self._monitored_metrics[key]
            return True
        return False

    def _metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for a metric."""
        labels_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{labels_str}"

    async def _detection_loop(self) -> None:
        """Background detection loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.check_interval)
                await self._run_detection()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")

    async def _run_detection(self) -> None:
        """Run anomaly detection on all monitored metrics."""
        for key, config in self._monitored_metrics.items():
            try:
                # Get current value
                value = self.metrics.get_current(config["name"], config["labels"])
                if value is None:
                    continue

                # Update state and detect
                state = self._metric_states.get(key)
                if state:
                    anomaly = self._detect_anomaly(state, value, config)
                    if anomaly:
                        self._record_anomaly(anomaly)

            except Exception as e:
                logger.error(f"Error detecting anomaly for {key}: {e}")

    def _detect_anomaly(
        self,
        state: MetricState,
        value: float,
        config: Dict[str, Any],
    ) -> Optional[Anomaly]:
        """Detect anomaly for a single metric value."""
        # Add value to history
        state.values.append((datetime.utcnow(), value))

        # Update min/max
        state.min_val = min(state.min_val, value)
        state.max_val = max(state.max_val, value)

        # Need enough samples
        if len(state.values) < config["min_samples"]:
            return None

        # Calculate statistics
        values = [v for _, v in state.values]
        state.mean = statistics.mean(values)
        state.std = statistics.stdev(values) if len(values) > 1 else 0.0

        # Update moving average
        recent = values[-self.config.moving_average_window:]
        state.moving_avg = statistics.mean(recent)
        state.moving_std = statistics.stdev(recent) if len(recent) > 1 else 0.0

        # Calculate trend
        if len(values) >= 10:
            first_half = statistics.mean(values[:len(values)//2])
            second_half = statistics.mean(values[len(values)//2:])
            state.trend = second_half - first_half

        # Detect using multiple methods
        anomaly = None

        # Method 1: Z-score
        z_score = self._calculate_z_score(value, state.mean, state.std)
        if abs(z_score) > config["z_score_threshold"] * self.config.sensitivity:
            anomaly = self._create_anomaly(
                state=state,
                value=value,
                anomaly_type=AnomalyType.SPIKE if z_score > 0 else AnomalyType.DROP,
                deviation=z_score,
                description=f"Z-score {z_score:.2f} exceeds threshold",
            )

        # Method 2: Moving average deviation
        if not anomaly and state.moving_std > 0:
            ma_deviation = (value - state.moving_avg) / state.moving_std
            if abs(ma_deviation) > config["z_score_threshold"] * 1.5:
                anomaly = self._create_anomaly(
                    state=state,
                    value=value,
                    anomaly_type=AnomalyType.OUTLIER,
                    deviation=ma_deviation,
                    description=f"Significant deviation from moving average",
                )

        # Method 3: Trend change detection
        if not anomaly and abs(state.trend) > state.std * 2:
            anomaly = self._create_anomaly(
                state=state,
                value=value,
                anomaly_type=AnomalyType.TREND_CHANGE,
                deviation=state.trend / state.std if state.std > 0 else 0,
                description=f"Significant trend change detected",
            )

        # Update consecutive anomalies
        if anomaly:
            state.consecutive_anomalies += 1
            state.last_anomaly_time = datetime.utcnow()
        else:
            state.consecutive_anomalies = 0

        return anomaly

    def _calculate_z_score(self, value: float, mean: float, std: float) -> float:
        """Calculate z-score."""
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _create_anomaly(
        self,
        state: MetricState,
        value: float,
        anomaly_type: AnomalyType,
        deviation: float,
        description: str,
    ) -> Anomaly:
        """Create an anomaly object."""
        return Anomaly(
            metric_name=state.name,
            labels=state.labels,
            anomaly_type=anomaly_type,
            expected_value=state.mean,
            actual_value=value,
            deviation=deviation,
            confidence=min(1.0, abs(deviation) / 5.0),  # Scale confidence
            description=description,
            possible_causes=self._infer_causes(state, anomaly_type, deviation),
        )

    def _infer_causes(
        self,
        state: MetricState,
        anomaly_type: AnomalyType,
        deviation: float,
    ) -> List[str]:
        """Infer possible causes for an anomaly."""
        causes = []

        if anomaly_type == AnomalyType.SPIKE:
            causes.append("Sudden increase in load")
            causes.append("Memory leak or resource exhaustion")
            causes.append("External dependency issue")

        elif anomaly_type == AnomalyType.DROP:
            causes.append("Service degradation")
            causes.append("Traffic drop")
            causes.append("Component failure")

        elif anomaly_type == AnomalyType.TREND_CHANGE:
            if state.trend > 0:
                causes.append("Gradual resource consumption")
                causes.append("Increasing traffic")
            else:
                causes.append("Decreasing activity")
                causes.append("Potential issue building up")

        elif anomaly_type == AnomalyType.OUTLIER:
            causes.append("One-time event")
            causes.append("Data quality issue")
            causes.append("Intermittent failure")

        return causes

    def _record_anomaly(self, anomaly: Anomaly) -> None:
        """Record and notify about an anomaly."""
        self._anomalies.append(anomaly)

        # Trim if needed
        if len(self._anomalies) > self._max_anomalies:
            self._anomalies = self._anomalies[-self._max_anomalies:]

        logger.warning(
            f"Anomaly detected: {anomaly.metric_name}",
            anomaly_type=anomaly.anomaly_type.value,
            deviation=anomaly.deviation,
            severity=anomaly.severity.value,
        )

        # Notify callbacks
        for callback in self._anomaly_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Anomaly callback error: {e}")

    # === Manual Detection ===

    def detect_in_values(
        self,
        values: List[float],
        metric_name: str = "custom",
    ) -> List[Anomaly]:
        """Detect anomalies in a list of values."""
        if len(values) < self.config.min_samples:
            return []

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        anomalies = []
        for i, value in enumerate(values):
            z_score = self._calculate_z_score(value, mean, std)
            if abs(z_score) > self.config.z_score_threshold:
                anomaly = Anomaly(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.SPIKE if z_score > 0 else AnomalyType.DROP,
                    expected_value=mean,
                    actual_value=value,
                    deviation=z_score,
                    confidence=min(1.0, abs(z_score) / 5.0),
                    description=f"Value at index {i} is anomalous (z={z_score:.2f})",
                )
                anomalies.append(anomaly)

        return anomalies

    # === Query Methods ===

    def get_anomalies(
        self,
        metric_name: str = None,
        since: datetime = None,
        severity: AlertSeverity = None,
        limit: int = 100,
    ) -> List[Anomaly]:
        """Get detected anomalies with optional filtering."""
        anomalies = self._anomalies

        if metric_name:
            anomalies = [a for a in anomalies if a.metric_name == metric_name]
        if since:
            anomalies = [a for a in anomalies if a.detected_at >= since]
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]

        return anomalies[-limit:]

    def get_metric_state(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> Optional[MetricState]:
        """Get state for a monitored metric."""
        key = self._metric_key(name, labels or {})
        return self._metric_states.get(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get anomaly detector statistics."""
        return {
            "monitored_metrics": len(self._monitored_metrics),
            "total_anomalies": len(self._anomalies),
            "anomalies_by_type": {
                t.value: len([a for a in self._anomalies if a.anomaly_type == t])
                for t in AnomalyType
            },
            "anomalies_by_severity": {
                s.value: len([a for a in self._anomalies if a.severity == s])
                for s in AlertSeverity
            },
        }


class IsolationForestDetector:
    """
    Isolation Forest based anomaly detection.

    Uses sklearn's IsolationForest for more sophisticated detection.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._model = None
        self._fitted = False

    def fit(self, values: List[float]) -> None:
        """Fit the model on historical values."""
        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np

            data = np.array(values).reshape(-1, 1)
            self._model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
            )
            self._model.fit(data)
            self._fitted = True

        except ImportError:
            logger.warning("sklearn not available for IsolationForest")

    def predict(self, value: float) -> bool:
        """Predict if a value is anomalous. Returns True if anomaly."""
        if not self._fitted or self._model is None:
            return False

        try:
            import numpy as np
            prediction = self._model.predict([[value]])
            return prediction[0] == -1  # -1 = anomaly
        except Exception:
            return False

    def score(self, value: float) -> float:
        """Get anomaly score for a value. Lower = more anomalous."""
        if not self._fitted or self._model is None:
            return 0.0

        try:
            import numpy as np
            return float(self._model.score_samples([[value]])[0])
        except Exception:
            return 0.0
