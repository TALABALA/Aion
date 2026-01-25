"""
Predictive Alerting for AIOps.

Implements proactive alerting capabilities:
- Forecast-based alerting (predict threshold violations)
- Trend-based alerting (detect degrading trends)
- Capacity prediction (forecast resource exhaustion)
- Anomaly forecasting (predict upcoming anomalies)
"""

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PredictiveAlert:
    """Alert generated from prediction."""
    alert_id: str
    metric_name: str
    predicted_value: float
    predicted_time: datetime
    current_value: float
    threshold: float
    severity: AlertSeverity
    confidence: PredictionConfidence
    time_to_violation: timedelta
    description: str
    recommendations: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Forecast:
    """Time series forecast result."""
    timestamps: List[datetime]
    values: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    confidence: float


@dataclass
class TrendInfo:
    """Information about a detected trend."""
    direction: str  # "increasing", "decreasing", "stable"
    slope: float
    r_squared: float
    is_significant: bool
    projected_value: float
    projection_time: datetime


# =============================================================================
# Trend Analysis
# =============================================================================

class TrendAnalyzer:
    """
    Analyze trends in time series data.

    Uses linear regression and statistical tests to detect
    significant trends.
    """

    def __init__(
        self,
        min_samples: int = 10,
        significance_threshold: float = 0.7
    ):
        self.min_samples = min_samples
        self.significance_threshold = significance_threshold

    def analyze(
        self,
        values: List[float],
        timestamps: List[datetime] = None
    ) -> TrendInfo:
        """Analyze trend in values."""
        if len(values) < self.min_samples:
            return TrendInfo(
                direction="stable",
                slope=0.0,
                r_squared=0.0,
                is_significant=False,
                projected_value=values[-1] if values else 0.0,
                projection_time=datetime.now()
            )

        # Convert timestamps to numeric
        if timestamps:
            x = [(t - timestamps[0]).total_seconds() for t in timestamps]
        else:
            x = list(range(len(values)))

        y = values

        # Linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        # Slope and intercept
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            slope = 0.0
            intercept = sum_y / n
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Determine direction
        if abs(slope) < 1e-10:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Project forward
        if timestamps:
            last_x = (timestamps[-1] - timestamps[0]).total_seconds()
            interval = last_x / (len(timestamps) - 1) if len(timestamps) > 1 else 60
            projection_x = last_x + interval * 10  # Project 10 intervals ahead
            projection_time = timestamps[-1] + timedelta(seconds=interval * 10)
        else:
            projection_x = len(values) + 10
            projection_time = datetime.now() + timedelta(minutes=10)

        projected_value = slope * projection_x + intercept

        return TrendInfo(
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            is_significant=r_squared > self.significance_threshold,
            projected_value=projected_value,
            projection_time=projection_time
        )

    def detect_changepoint(
        self,
        values: List[float],
        window_size: int = 20
    ) -> Optional[int]:
        """Detect changepoint in time series."""
        if len(values) < window_size * 2:
            return None

        best_score = 0.0
        best_point = None

        for i in range(window_size, len(values) - window_size):
            left = values[i - window_size:i]
            right = values[i:i + window_size]

            left_mean = sum(left) / len(left)
            right_mean = sum(right) / len(right)

            # Score based on mean difference
            left_std = math.sqrt(sum((v - left_mean)**2 for v in left) / len(left))
            right_std = math.sqrt(sum((v - right_mean)**2 for v in right) / len(right))

            pooled_std = math.sqrt((left_std**2 + right_std**2) / 2) + 1e-8
            score = abs(right_mean - left_mean) / pooled_std

            if score > best_score:
                best_score = score
                best_point = i

        # Return changepoint if significant
        if best_score > 2.0:  # ~2 standard deviations
            return best_point

        return None


# =============================================================================
# Anomaly Forecasting
# =============================================================================

class AnomalyForecaster:
    """
    Forecast upcoming anomalies.

    Uses pattern recognition and predictive models to
    anticipate anomalies before they occur.
    """

    def __init__(
        self,
        forecast_horizon: int = 30,
        anomaly_threshold: float = 0.7
    ):
        self.forecast_horizon = forecast_horizon
        self.anomaly_threshold = anomaly_threshold

        # Historical patterns
        self._patterns: Dict[str, List[Tuple[List[float], bool]]] = {}
        self._seasonal_profiles: Dict[str, Dict[int, float]] = {}

    def learn_pattern(
        self,
        metric_name: str,
        values: List[float],
        had_anomaly: bool
    ):
        """Learn from historical pattern."""
        if metric_name not in self._patterns:
            self._patterns[metric_name] = []

        # Store pattern
        self._patterns[metric_name].append((values[-20:], had_anomaly))

        # Keep only recent patterns
        if len(self._patterns[metric_name]) > 1000:
            self._patterns[metric_name] = self._patterns[metric_name][-500:]

    def learn_seasonality(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime]
    ):
        """Learn seasonal patterns."""
        if metric_name not in self._seasonal_profiles:
            self._seasonal_profiles[metric_name] = {}

        # Build hourly profile
        hourly_values: Dict[int, List[float]] = {}
        for v, t in zip(values, timestamps):
            hour = t.hour
            if hour not in hourly_values:
                hourly_values[hour] = []
            hourly_values[hour].append(v)

        for hour, vals in hourly_values.items():
            self._seasonal_profiles[metric_name][hour] = sum(vals) / len(vals)

    def forecast_anomaly_probability(
        self,
        metric_name: str,
        current_values: List[float],
        current_time: datetime
    ) -> Tuple[float, str]:
        """
        Forecast probability of anomaly in near future.

        Returns:
            (probability, explanation)
        """
        probability = 0.0
        reasons = []

        # Pattern matching
        if metric_name in self._patterns:
            pattern_prob = self._match_patterns(current_values)
            if pattern_prob > 0.5:
                probability = max(probability, pattern_prob)
                reasons.append(f"Similar to historical anomaly patterns ({pattern_prob:.0%})")

        # Trend analysis
        trend_analyzer = TrendAnalyzer()
        trend = trend_analyzer.analyze(current_values)

        if trend.is_significant and trend.direction != "stable":
            trend_prob = min(0.9, trend.r_squared)
            probability = max(probability, trend_prob)
            reasons.append(f"Strong {trend.direction} trend detected (R²={trend.r_squared:.2f})")

        # Seasonal deviation
        if metric_name in self._seasonal_profiles:
            seasonal_prob = self._check_seasonal_deviation(
                metric_name, current_values[-1], current_time
            )
            if seasonal_prob > 0.5:
                probability = max(probability, seasonal_prob)
                reasons.append(f"Deviation from seasonal pattern ({seasonal_prob:.0%})")

        # Volatility spike
        if len(current_values) > 10:
            recent_std = math.sqrt(
                sum((v - sum(current_values[-10:]) / 10)**2
                    for v in current_values[-10:]) / 10
            )
            historical_std = math.sqrt(
                sum((v - sum(current_values) / len(current_values))**2
                    for v in current_values) / len(current_values)
            )

            if historical_std > 0 and recent_std > historical_std * 2:
                volatility_prob = min(0.8, recent_std / historical_std / 3)
                probability = max(probability, volatility_prob)
                reasons.append(f"Increased volatility detected")

        explanation = "; ".join(reasons) if reasons else "No anomaly indicators"
        return (probability, explanation)

    def _match_patterns(self, current_values: List[float]) -> float:
        """Match current pattern against historical anomaly patterns."""
        if not current_values:
            return 0.0

        pattern = current_values[-20:] if len(current_values) >= 20 else current_values

        # Normalize pattern
        mean = sum(pattern) / len(pattern)
        std = math.sqrt(sum((v - mean)**2 for v in pattern) / len(pattern)) + 1e-8
        normalized = [(v - mean) / std for v in pattern]

        best_similarity = 0.0
        anomaly_matches = 0
        total_matches = 0

        for metric_patterns in self._patterns.values():
            for hist_pattern, had_anomaly in metric_patterns:
                if len(hist_pattern) < 5:
                    continue

                # Normalize historical pattern
                h_mean = sum(hist_pattern) / len(hist_pattern)
                h_std = math.sqrt(
                    sum((v - h_mean)**2 for v in hist_pattern) / len(hist_pattern)
                ) + 1e-8
                h_normalized = [(v - h_mean) / h_std for v in hist_pattern]

                # Compute similarity (DTW-like)
                similarity = self._pattern_similarity(normalized, h_normalized)

                if similarity > 0.7:
                    total_matches += 1
                    if had_anomaly:
                        anomaly_matches += 1

                best_similarity = max(best_similarity, similarity)

        if total_matches > 0:
            return anomaly_matches / total_matches * best_similarity

        return 0.0

    def _pattern_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute similarity between two patterns."""
        # Use correlation coefficient
        if len(a) < 2 or len(b) < 2:
            return 0.0

        # Resample to same length
        min_len = min(len(a), len(b))
        a = a[-min_len:]
        b = b[-min_len:]

        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)

        cov = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b)) / len(a)
        std_a = math.sqrt(sum((ai - mean_a)**2 for ai in a) / len(a))
        std_b = math.sqrt(sum((bi - mean_b)**2 for bi in b) / len(b))

        if std_a * std_b == 0:
            return 0.0

        correlation = cov / (std_a * std_b)
        return max(0, correlation)

    def _check_seasonal_deviation(
        self,
        metric_name: str,
        current_value: float,
        current_time: datetime
    ) -> float:
        """Check deviation from seasonal pattern."""
        hour = current_time.hour
        profile = self._seasonal_profiles.get(metric_name, {})

        if hour not in profile:
            return 0.0

        expected = profile[hour]
        if expected == 0:
            return 0.0

        deviation = abs(current_value - expected) / abs(expected)
        return min(1.0, deviation / 2)


# =============================================================================
# Capacity Prediction
# =============================================================================

class CapacityPredictor:
    """
    Predict resource capacity exhaustion.

    Forecasts when resources will reach critical thresholds
    based on usage trends.
    """

    def __init__(
        self,
        default_thresholds: Dict[str, float] = None
    ):
        self.thresholds = default_thresholds or {
            "cpu": 0.90,
            "memory": 0.85,
            "disk": 0.90,
            "connections": 0.95,
        }

        self._history: Dict[str, deque] = {}
        self._capacity: Dict[str, float] = {}

    def set_capacity(self, metric_name: str, capacity: float):
        """Set the total capacity for a metric."""
        self._capacity[metric_name] = capacity

    def set_threshold(self, metric_name: str, threshold: float):
        """Set critical threshold for a metric."""
        self.thresholds[metric_name] = threshold

    def update(self, metric_name: str, value: float, timestamp: datetime = None):
        """Update with new metric value."""
        timestamp = timestamp or datetime.now()

        if metric_name not in self._history:
            self._history[metric_name] = deque(maxlen=1000)

        self._history[metric_name].append((timestamp, value))

    def predict_exhaustion(
        self,
        metric_name: str,
        capacity: float = None
    ) -> Optional[Tuple[datetime, float, float]]:
        """
        Predict when capacity will be exhausted.

        Returns:
            (exhaustion_time, current_usage_pct, growth_rate) or None
        """
        if metric_name not in self._history:
            return None

        history = list(self._history[metric_name])
        if len(history) < 10:
            return None

        capacity = capacity or self._capacity.get(metric_name, 100.0)
        threshold = self.thresholds.get(metric_name, 0.9)

        # Extract values and times
        timestamps = [t for t, v in history]
        values = [v for t, v in history]

        # Analyze trend
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze(values, timestamps)

        if not trend.is_significant or trend.slope <= 0:
            return None  # No concerning trend

        # Current usage
        current_value = values[-1]
        current_pct = current_value / capacity

        # Calculate time to threshold
        threshold_value = capacity * threshold
        remaining = threshold_value - current_value

        if remaining <= 0:
            # Already exceeded
            return (datetime.now(), current_pct, trend.slope)

        # Time to exhaustion (in same units as timestamps)
        if timestamps:
            time_unit = (timestamps[-1] - timestamps[0]).total_seconds() / (len(timestamps) - 1)
            steps_to_exhaustion = remaining / trend.slope
            time_to_exhaustion = timedelta(seconds=steps_to_exhaustion * time_unit)
            exhaustion_time = timestamps[-1] + time_to_exhaustion
        else:
            exhaustion_time = datetime.now() + timedelta(hours=remaining / trend.slope)

        return (exhaustion_time, current_pct, trend.slope)

    def get_forecast(
        self,
        metric_name: str,
        horizon: timedelta = timedelta(hours=24)
    ) -> Optional[Forecast]:
        """Get capacity forecast for metric."""
        if metric_name not in self._history:
            return None

        history = list(self._history[metric_name])
        if len(history) < 10:
            return None

        timestamps = [t for t, v in history]
        values = [v for t, v in history]

        # Analyze trend
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze(values, timestamps)

        # Generate forecast
        last_time = timestamps[-1]
        interval = (timestamps[-1] - timestamps[0]).total_seconds() / (len(timestamps) - 1)

        forecast_timestamps = []
        forecast_values = []

        # Simple linear extrapolation
        for i in range(int(horizon.total_seconds() / interval)):
            t = last_time + timedelta(seconds=interval * (i + 1))
            v = values[-1] + trend.slope * (i + 1)
            forecast_timestamps.append(t)
            forecast_values.append(v)

        # Confidence bounds based on historical variance
        variance = sum((v - sum(values) / len(values))**2 for v in values) / len(values)
        std = math.sqrt(variance)

        lower_bound = [v - 1.96 * std for v in forecast_values]
        upper_bound = [v + 1.96 * std for v in forecast_values]

        return Forecast(
            timestamps=forecast_timestamps,
            values=forecast_values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence=trend.r_squared
        )


# =============================================================================
# Alert Predictor
# =============================================================================

class AlertPredictor:
    """
    Predict upcoming alerts based on current metrics.

    Combines trend analysis, pattern matching, and threshold
    projection to forecast alerts before they occur.
    """

    def __init__(
        self,
        prediction_horizon: timedelta = timedelta(hours=1),
        confidence_threshold: float = 0.6
    ):
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold

        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_forecaster = AnomalyForecaster()
        self.capacity_predictor = CapacityPredictor()

        # Alert thresholds
        self._thresholds: Dict[str, Tuple[float, float]] = {}  # metric -> (warning, critical)

    def set_threshold(
        self,
        metric_name: str,
        warning: float,
        critical: float
    ):
        """Set alert thresholds for a metric."""
        self._thresholds[metric_name] = (warning, critical)

    def predict(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime] = None
    ) -> List[PredictiveAlert]:
        """
        Predict upcoming alerts for a metric.

        Returns list of predicted alerts with confidence scores.
        """
        alerts = []
        now = datetime.now()

        if not values:
            return alerts

        current_value = values[-1]
        timestamps = timestamps or [now - timedelta(minutes=i)
                                    for i in range(len(values) - 1, -1, -1)]

        # Get thresholds
        warning_threshold, critical_threshold = self._thresholds.get(
            metric_name, (float('inf'), float('inf'))
        )

        # Trend-based prediction
        trend = self.trend_analyzer.analyze(values, timestamps)

        if trend.is_significant:
            # Project when thresholds will be crossed
            if trend.direction == "increasing":
                # Time to warning threshold
                if current_value < warning_threshold:
                    time_to_warning = self._time_to_threshold(
                        current_value, warning_threshold, trend.slope, timestamps
                    )
                    if time_to_warning and time_to_warning < self.prediction_horizon:
                        alerts.append(self._create_alert(
                            metric_name=metric_name,
                            predicted_value=warning_threshold,
                            current_value=current_value,
                            threshold=warning_threshold,
                            time_to_violation=time_to_warning,
                            severity=AlertSeverity.WARNING,
                            confidence=self._confidence_from_r2(trend.r_squared),
                            reason=f"Increasing trend will reach warning threshold"
                        ))

                # Time to critical threshold
                if current_value < critical_threshold:
                    time_to_critical = self._time_to_threshold(
                        current_value, critical_threshold, trend.slope, timestamps
                    )
                    if time_to_critical and time_to_critical < self.prediction_horizon:
                        alerts.append(self._create_alert(
                            metric_name=metric_name,
                            predicted_value=critical_threshold,
                            current_value=current_value,
                            threshold=critical_threshold,
                            time_to_violation=time_to_critical,
                            severity=AlertSeverity.CRITICAL,
                            confidence=self._confidence_from_r2(trend.r_squared),
                            reason=f"Increasing trend will reach critical threshold"
                        ))

        # Anomaly-based prediction
        anomaly_prob, anomaly_reason = self.anomaly_forecaster.forecast_anomaly_probability(
            metric_name, values, now
        )

        if anomaly_prob > self.confidence_threshold:
            alerts.append(self._create_alert(
                metric_name=metric_name,
                predicted_value=current_value * 1.5,  # Estimate
                current_value=current_value,
                threshold=warning_threshold,
                time_to_violation=timedelta(minutes=15),  # Estimate
                severity=AlertSeverity.WARNING,
                confidence=self._confidence_from_probability(anomaly_prob),
                reason=f"Anomaly predicted: {anomaly_reason}"
            ))

        # Capacity-based prediction
        self.capacity_predictor.update(metric_name, current_value, now)
        exhaustion = self.capacity_predictor.predict_exhaustion(metric_name)

        if exhaustion:
            exhaustion_time, current_pct, growth_rate = exhaustion
            time_to_exhaustion = exhaustion_time - now

            if time_to_exhaustion < self.prediction_horizon:
                alerts.append(self._create_alert(
                    metric_name=metric_name,
                    predicted_value=current_value * 1.1,
                    current_value=current_value,
                    threshold=critical_threshold,
                    time_to_violation=time_to_exhaustion,
                    severity=AlertSeverity.CRITICAL,
                    confidence=PredictionConfidence.HIGH,
                    reason=f"Capacity exhaustion predicted at {current_pct:.0%} utilization"
                ))

        # Filter by confidence
        alerts = [a for a in alerts
                 if self._confidence_to_value(a.confidence) >= self.confidence_threshold]

        return alerts

    def _time_to_threshold(
        self,
        current: float,
        threshold: float,
        slope: float,
        timestamps: List[datetime]
    ) -> Optional[timedelta]:
        """Calculate time to reach threshold."""
        if slope <= 0:
            return None

        steps = (threshold - current) / slope

        if timestamps and len(timestamps) > 1:
            interval = (timestamps[-1] - timestamps[0]).total_seconds() / (len(timestamps) - 1)
            return timedelta(seconds=steps * interval)

        return timedelta(minutes=steps)

    def _confidence_from_r2(self, r_squared: float) -> PredictionConfidence:
        """Convert R² to confidence level."""
        if r_squared > 0.9:
            return PredictionConfidence.HIGH
        elif r_squared > 0.7:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW

    def _confidence_from_probability(self, prob: float) -> PredictionConfidence:
        """Convert probability to confidence level."""
        if prob > 0.8:
            return PredictionConfidence.HIGH
        elif prob > 0.6:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW

    def _confidence_to_value(self, confidence: PredictionConfidence) -> float:
        """Convert confidence level to numeric value."""
        return {
            PredictionConfidence.LOW: 0.4,
            PredictionConfidence.MEDIUM: 0.7,
            PredictionConfidence.HIGH: 0.9
        }[confidence]

    def _create_alert(
        self,
        metric_name: str,
        predicted_value: float,
        current_value: float,
        threshold: float,
        time_to_violation: timedelta,
        severity: AlertSeverity,
        confidence: PredictionConfidence,
        reason: str
    ) -> PredictiveAlert:
        """Create a predictive alert."""
        import uuid

        recommendations = self._generate_recommendations(
            metric_name, severity, reason
        )

        return PredictiveAlert(
            alert_id=str(uuid.uuid4())[:8],
            metric_name=metric_name,
            predicted_value=predicted_value,
            predicted_time=datetime.now() + time_to_violation,
            current_value=current_value,
            threshold=threshold,
            severity=severity,
            confidence=confidence,
            time_to_violation=time_to_violation,
            description=f"Predicted {severity.value} for {metric_name}: {reason}",
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        metric_name: str,
        severity: AlertSeverity,
        reason: str
    ) -> List[str]:
        """Generate recommendations for predicted alert."""
        recommendations = []

        # Generic recommendations
        recommendations.append(f"Monitor {metric_name} closely")

        if severity == AlertSeverity.CRITICAL:
            recommendations.append("Consider scaling resources proactively")
            recommendations.append("Review recent changes that may have caused this trend")

        if "capacity" in reason.lower():
            recommendations.append("Plan capacity expansion")
            recommendations.append("Review resource allocation")

        if "anomaly" in reason.lower():
            recommendations.append("Check for unusual traffic patterns")
            recommendations.append("Review recent deployments")

        return recommendations


# =============================================================================
# Predictive Alerter
# =============================================================================

class PredictiveAlerter:
    """
    Main predictive alerting engine.

    Continuously monitors metrics and generates predictive alerts
    before issues occur.
    """

    def __init__(
        self,
        prediction_horizon: timedelta = timedelta(hours=1),
        check_interval: timedelta = timedelta(minutes=5),
        min_data_points: int = 20
    ):
        self.prediction_horizon = prediction_horizon
        self.check_interval = check_interval
        self.min_data_points = min_data_points

        self.predictor = AlertPredictor(prediction_horizon)

        # Metric history
        self._metrics: Dict[str, deque] = {}

        # Alert handlers
        self._handlers: List[Callable[[PredictiveAlert], None]] = []

        # Deduplication
        self._recent_alerts: Dict[str, datetime] = {}
        self._dedup_window = timedelta(minutes=30)

    def add_handler(self, handler: Callable[[PredictiveAlert], None]):
        """Add alert handler."""
        self._handlers.append(handler)

    def configure_metric(
        self,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        capacity: float = None
    ):
        """Configure a metric for predictive alerting."""
        self.predictor.set_threshold(metric_name, warning_threshold, critical_threshold)
        if capacity:
            self.predictor.capacity_predictor.set_capacity(metric_name, capacity)

    def observe(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime = None
    ):
        """Observe a metric value."""
        timestamp = timestamp or datetime.now()

        if metric_name not in self._metrics:
            self._metrics[metric_name] = deque(maxlen=1000)

        self._metrics[metric_name].append((timestamp, value))

    def check(self, metric_name: str = None) -> List[PredictiveAlert]:
        """
        Check for predicted alerts.

        Args:
            metric_name: Specific metric to check, or None for all
        """
        alerts = []

        metrics_to_check = [metric_name] if metric_name else list(self._metrics.keys())

        for name in metrics_to_check:
            if name not in self._metrics:
                continue

            history = list(self._metrics[name])
            if len(history) < self.min_data_points:
                continue

            timestamps = [t for t, v in history]
            values = [v for t, v in history]

            # Get predictions
            predicted_alerts = self.predictor.predict(name, values, timestamps)

            # Deduplicate
            for alert in predicted_alerts:
                if self._should_send_alert(alert):
                    alerts.append(alert)
                    self._notify_handlers(alert)

        return alerts

    def _should_send_alert(self, alert: PredictiveAlert) -> bool:
        """Check if alert should be sent (deduplication)."""
        key = f"{alert.metric_name}:{alert.severity.value}"

        if key in self._recent_alerts:
            last_sent = self._recent_alerts[key]
            if datetime.now() - last_sent < self._dedup_window:
                return False

        self._recent_alerts[key] = datetime.now()
        return True

    def _notify_handlers(self, alert: PredictiveAlert):
        """Notify all handlers of an alert."""
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
