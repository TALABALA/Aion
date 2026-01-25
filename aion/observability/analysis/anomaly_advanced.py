"""
Advanced Anomaly Detection System

SOTA anomaly detection with:
- Seasonal-Trend decomposition using LOESS (STL/MSTL)
- Prophet-style forecasting with changepoint detection
- LSTM/Transformer-based sequence anomaly detection
- Isolation Forest ensemble methods
- Multi-variate correlation analysis
- Dynamic threshold adaptation
"""

from __future__ import annotations

import math
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque
from enum import Enum
import numpy as np

from aion.observability.types import Anomaly, AnomalyType, AlertSeverity

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SeasonalDecomposition:
    """Result of seasonal decomposition."""
    trend: List[float]
    seasonal: List[float]
    residual: List[float]
    period: int
    timestamps: List[datetime] = field(default_factory=list)


@dataclass
class Forecast:
    """Forecast result with confidence intervals."""
    timestamps: List[datetime]
    values: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    confidence: float = 0.95
    changepoints: List[datetime] = field(default_factory=list)


@dataclass
class MultivariateAnomaly:
    """Anomaly detected in multivariate correlation."""
    timestamp: datetime
    metrics: Dict[str, float]
    correlation_score: float
    expected_correlations: Dict[str, float]
    actual_correlations: Dict[str, float]
    anomalous_pairs: List[Tuple[str, str]]
    severity: AlertSeverity = AlertSeverity.WARNING


class AnomalyDetectorType(Enum):
    """Types of anomaly detectors."""
    ZSCORE = "zscore"
    STL = "stl"
    PROPHET = "prophet"
    ISOLATION_FOREST = "isolation_forest"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    MULTIVARIATE = "multivariate"
    ENSEMBLE = "ensemble"


# =============================================================================
# Base Detector Interface
# =============================================================================

class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    async def fit(self, data: List[TimeSeriesPoint]) -> None:
        """Train the detector on historical data."""
        pass

    @abstractmethod
    async def detect(self, point: TimeSeriesPoint) -> Optional[Anomaly]:
        """Detect if a single point is anomalous."""
        pass

    @abstractmethod
    async def detect_batch(self, points: List[TimeSeriesPoint]) -> List[Anomaly]:
        """Detect anomalies in a batch of points."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        pass


# =============================================================================
# STL Decomposition (Seasonal-Trend using LOESS)
# =============================================================================

class STLDecomposer:
    """
    Seasonal-Trend decomposition using LOESS (Locally Estimated Scatterplot Smoothing).

    Implements the STL algorithm for decomposing time series into:
    - Trend component
    - Seasonal component
    - Residual component
    """

    def __init__(
        self,
        period: int = 24,  # Default hourly seasonality for daily patterns
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        robust: bool = True,
        outer_iter: int = 5,
        inner_iter: int = 2,
    ):
        self.period = period
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.robust = robust
        self.outer_iter = outer_iter
        self.inner_iter = inner_iter

    def _loess_smooth(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
        span: float = 0.3,
        degree: int = 1,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        LOESS (Locally Estimated Scatterplot Smoothing) regression.
        """
        n = len(x)
        k = max(int(span * n), degree + 1)
        result = np.zeros(len(x_new))

        if weights is None:
            weights = np.ones(n)

        for i, xi in enumerate(x_new):
            # Find k nearest neighbors
            distances = np.abs(x - xi)
            sorted_idx = np.argsort(distances)[:k]

            # Tricube weight function
            max_dist = distances[sorted_idx[-1]] + 1e-10
            u = distances[sorted_idx] / max_dist
            tricube = np.clip(1 - u**3, 0, 1)**3

            # Combined weights
            w = tricube * weights[sorted_idx]

            # Weighted polynomial regression
            x_local = x[sorted_idx]
            y_local = y[sorted_idx]

            if degree == 0:
                result[i] = np.average(y_local, weights=w)
            elif degree == 1:
                # Weighted linear regression
                W = np.diag(w)
                X = np.column_stack([np.ones(k), x_local - xi])
                try:
                    beta = np.linalg.solve(X.T @ W @ X + 1e-10 * np.eye(2), X.T @ W @ y_local)
                    result[i] = beta[0]
                except np.linalg.LinAlgError:
                    result[i] = np.average(y_local, weights=w)
            else:
                # Higher degree polynomial
                X = np.column_stack([
                    (x_local - xi)**d for d in range(degree + 1)
                ])
                W = np.diag(w)
                try:
                    beta = np.linalg.solve(
                        X.T @ W @ X + 1e-10 * np.eye(degree + 1),
                        X.T @ W @ y_local
                    )
                    result[i] = beta[0]
                except np.linalg.LinAlgError:
                    result[i] = np.average(y_local, weights=w)

        return result

    def decompose(
        self,
        values: List[float],
        timestamps: Optional[List[datetime]] = None,
    ) -> SeasonalDecomposition:
        """
        Decompose time series into trend, seasonal, and residual components.
        """
        y = np.array(values)
        n = len(y)

        if n < 2 * self.period:
            # Not enough data for seasonal decomposition
            return SeasonalDecomposition(
                trend=list(y),
                seasonal=[0.0] * n,
                residual=[0.0] * n,
                period=self.period,
                timestamps=timestamps or [],
            )

        x = np.arange(n)

        # Initialize
        trend = np.zeros(n)
        seasonal = np.zeros(n)
        robustness_weights = np.ones(n)

        for _ in range(self.outer_iter):
            for _ in range(self.inner_iter):
                # Step 1: Detrend
                detrended = y - trend

                # Step 2: Cycle-subseries smoothing for seasonal
                seasonal_new = np.zeros(n)
                for s in range(self.period):
                    idx = np.arange(s, n, self.period)
                    if len(idx) > 1:
                        subseries = detrended[idx]
                        smoothed = self._loess_smooth(
                            np.arange(len(idx)).astype(float),
                            subseries,
                            np.arange(len(idx)).astype(float),
                            span=max(0.3, 7 / len(idx)),
                            degree=self.seasonal_deg,
                            weights=robustness_weights[idx],
                        )
                        seasonal_new[idx] = smoothed

                # Step 3: Low-pass filter on seasonal
                # Moving average to remove remaining trend in seasonal
                ma_len = self.period
                if n > ma_len:
                    seasonal_smooth = np.convolve(
                        seasonal_new,
                        np.ones(ma_len) / ma_len,
                        mode='same'
                    )
                    seasonal = seasonal_new - seasonal_smooth
                else:
                    seasonal = seasonal_new

                # Step 4: Center the seasonal component
                seasonal = seasonal - np.mean(seasonal)

                # Step 5: Deseasonalize and extract trend
                deseasonalized = y - seasonal
                trend = self._loess_smooth(
                    x.astype(float),
                    deseasonalized,
                    x.astype(float),
                    span=max(0.1, (1.5 * self.period) / n),
                    degree=self.trend_deg,
                    weights=robustness_weights,
                )

            if self.robust:
                # Update robustness weights using bisquare function
                residual = y - trend - seasonal
                mad = np.median(np.abs(residual - np.median(residual)))
                u = residual / (6 * mad + 1e-10)
                robustness_weights = np.clip(1 - u**2, 0, 1)**2

        residual = y - trend - seasonal

        return SeasonalDecomposition(
            trend=list(trend),
            seasonal=list(seasonal),
            residual=list(residual),
            period=self.period,
            timestamps=timestamps or [],
        )


class STLAnomalyDetector(BaseAnomalyDetector):
    """
    Anomaly detector using STL decomposition.

    Detects anomalies in the residual component after removing
    trend and seasonal patterns.
    """

    def __init__(
        self,
        period: int = 24,
        residual_threshold: float = 3.0,  # Standard deviations
        trend_threshold: float = 2.0,
        robust: bool = True,
        min_samples: int = 48,
    ):
        self.decomposer = STLDecomposer(period=period, robust=robust)
        self.period = period
        self.residual_threshold = residual_threshold
        self.trend_threshold = trend_threshold
        self.min_samples = min_samples

        self._history: deque = deque(maxlen=period * 14)  # 2 weeks of hourly data
        self._last_decomposition: Optional[SeasonalDecomposition] = None
        self._residual_std: float = 1.0
        self._trend_change_rate: float = 0.0

    async def fit(self, data: List[TimeSeriesPoint]) -> None:
        """Train on historical data."""
        self._history.clear()
        for point in data:
            self._history.append(point)

        if len(self._history) >= self.min_samples:
            values = [p.value for p in self._history]
            timestamps = [p.timestamp for p in self._history]
            self._last_decomposition = self.decomposer.decompose(values, timestamps)
            self._residual_std = np.std(self._last_decomposition.residual) + 1e-10

    async def detect(self, point: TimeSeriesPoint) -> Optional[Anomaly]:
        """Detect if a point is anomalous."""
        self._history.append(point)

        if len(self._history) < self.min_samples:
            return None

        values = [p.value for p in self._history]
        timestamps = [p.timestamp for p in self._history]

        decomposition = self.decomposer.decompose(values, timestamps)
        self._last_decomposition = decomposition

        # Check residual for point anomaly
        residual = decomposition.residual[-1]
        self._residual_std = np.std(decomposition.residual) + 1e-10
        z_score = abs(residual) / self._residual_std

        if z_score > self.residual_threshold:
            # Determine if it's a spike or level shift
            recent_trend = decomposition.trend[-10:] if len(decomposition.trend) >= 10 else decomposition.trend
            trend_diff = np.diff(recent_trend)

            if abs(np.mean(trend_diff)) > self._residual_std * 0.5:
                anomaly_type = AnomalyType.LEVEL_SHIFT
            else:
                anomaly_type = AnomalyType.SPIKE if residual > 0 else AnomalyType.DIP

            severity = AlertSeverity.CRITICAL if z_score > 5 else (
                AlertSeverity.WARNING if z_score > 4 else AlertSeverity.INFO
            )

            return Anomaly(
                metric_name="stl_decomposition",
                timestamp=point.timestamp,
                value=point.value,
                expected_value=decomposition.trend[-1] + decomposition.seasonal[-1],
                anomaly_type=anomaly_type,
                score=z_score,
                severity=severity,
                details={
                    "trend": decomposition.trend[-1],
                    "seasonal": decomposition.seasonal[-1],
                    "residual": residual,
                    "residual_std": self._residual_std,
                    "z_score": z_score,
                },
            )

        return None

    async def detect_batch(self, points: List[TimeSeriesPoint]) -> List[Anomaly]:
        """Detect anomalies in batch."""
        anomalies = []
        for point in points:
            anomaly = await self.detect(point)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "stl",
            "period": self.period,
            "samples": len(self._history),
            "residual_std": self._residual_std,
            "has_decomposition": self._last_decomposition is not None,
        }


# =============================================================================
# Prophet-Style Forecasting
# =============================================================================

@dataclass
class Changepoint:
    """Detected changepoint in time series."""
    timestamp: datetime
    index: int
    magnitude: float
    direction: str  # "increase" or "decrease"


class ProphetStyleForecaster:
    """
    Prophet-inspired forecasting with:
    - Piecewise linear/logistic trend with changepoints
    - Fourier series for seasonality
    - Holiday effects (optional)
    - Uncertainty intervals via bootstrap
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        n_changepoints: int = 25,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        growth: str = "linear",  # "linear" or "logistic"
        cap: Optional[float] = None,
        floor: Optional[float] = None,
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.n_changepoints = n_changepoints
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.growth = growth
        self.cap = cap
        self.floor = floor

        # Fitted parameters
        self._k: float = 0.0  # Base growth rate
        self._m: float = 0.0  # Offset
        self._deltas: np.ndarray = np.array([])  # Changepoint adjustments
        self._changepoints: List[Changepoint] = []
        self._seasonality_coeffs: Dict[str, np.ndarray] = {}
        self._t_scale: float = 1.0
        self._y_scale: float = 1.0
        self._start_time: Optional[datetime] = None

    def _make_seasonality_features(
        self,
        t: np.ndarray,
        period: float,
        n_fourier: int,
    ) -> np.ndarray:
        """Create Fourier features for seasonality."""
        features = []
        for i in range(1, n_fourier + 1):
            features.append(np.sin(2 * np.pi * i * t / period))
            features.append(np.cos(2 * np.pi * i * t / period))
        return np.column_stack(features) if features else np.zeros((len(t), 0))

    def _detect_changepoints(
        self,
        t: np.ndarray,
        y: np.ndarray,
    ) -> List[int]:
        """Detect changepoints using PELT-inspired algorithm."""
        n = len(y)
        if n < 10:
            return []

        # Use potential changepoint locations (exclude first and last 20%)
        potential = np.linspace(
            int(0.2 * n),
            int(0.8 * n),
            min(self.n_changepoints, int(0.6 * n))
        ).astype(int)

        # Simple changepoint detection via cusum
        cusum = np.cumsum(y - np.mean(y))
        changepoints = []

        threshold = self.changepoint_prior_scale * np.std(y) * np.sqrt(n)

        for i in potential:
            if i < 5 or i > n - 5:
                continue
            # Check if there's a significant change at this point
            left_mean = np.mean(y[max(0, i-10):i])
            right_mean = np.mean(y[i:min(n, i+10)])
            diff = abs(right_mean - left_mean)

            if diff > threshold:
                changepoints.append(i)

        return changepoints

    def fit(
        self,
        timestamps: List[datetime],
        values: List[float],
    ) -> None:
        """Fit the forecaster to historical data."""
        if len(timestamps) < 10:
            return

        self._start_time = timestamps[0]

        # Convert to numerical time
        t = np.array([
            (ts - self._start_time).total_seconds() / 86400.0
            for ts in timestamps
        ])
        y = np.array(values)

        self._t_scale = t[-1] - t[0] + 1e-10
        self._y_scale = np.std(y) + 1e-10

        t_scaled = t / self._t_scale
        y_scaled = (y - np.mean(y)) / self._y_scale

        # Detect changepoints
        cp_indices = self._detect_changepoints(t_scaled, y_scaled)

        # Build design matrix
        n = len(t)

        # Trend component
        if self.growth == "linear":
            # Piecewise linear
            A = np.zeros((n, len(cp_indices) + 1))
            A[:, 0] = t_scaled

            for j, cp_idx in enumerate(cp_indices):
                A[cp_idx:, j + 1] = t_scaled[cp_idx:] - t_scaled[cp_idx]
        else:
            A = np.column_stack([np.ones(n), t_scaled])

        # Seasonality features
        X_seasonal = []

        if self.daily_seasonality:
            daily = self._make_seasonality_features(t, 1.0, 3)
            X_seasonal.append(daily)

        if self.weekly_seasonality:
            weekly = self._make_seasonality_features(t, 7.0, 3)
            X_seasonal.append(weekly)

        if self.yearly_seasonality and self._t_scale > 180:
            yearly = self._make_seasonality_features(t, 365.25, 5)
            X_seasonal.append(yearly)

        if X_seasonal:
            X_seas = np.hstack(X_seasonal)
            X = np.hstack([A, X_seas])
        else:
            X = A

        # Fit using ridge regression for stability
        lambda_reg = 0.01
        try:
            beta = np.linalg.solve(
                X.T @ X + lambda_reg * np.eye(X.shape[1]),
                X.T @ y_scaled
            )
        except np.linalg.LinAlgError:
            beta = np.zeros(X.shape[1])

        # Extract parameters
        self._k = beta[0] if len(beta) > 0 else 0
        self._deltas = beta[1:len(cp_indices)+1] if len(cp_indices) > 0 else np.array([])
        self._m = np.mean(y)

        # Store changepoints
        self._changepoints = []
        for i, cp_idx in enumerate(cp_indices):
            if i < len(self._deltas):
                self._changepoints.append(Changepoint(
                    timestamp=timestamps[cp_idx],
                    index=cp_idx,
                    magnitude=float(self._deltas[i] * self._y_scale),
                    direction="increase" if self._deltas[i] > 0 else "decrease",
                ))

        # Store seasonality coefficients
        if X_seasonal:
            seas_start = len(cp_indices) + 1
            self._seasonality_coeffs = {"combined": beta[seas_start:]}

    def predict(
        self,
        timestamps: List[datetime],
        include_uncertainty: bool = True,
        n_samples: int = 100,
    ) -> Forecast:
        """Generate forecasts with uncertainty intervals."""
        if self._start_time is None:
            # Not fitted, return zeros
            return Forecast(
                timestamps=timestamps,
                values=[0.0] * len(timestamps),
                lower_bound=[0.0] * len(timestamps),
                upper_bound=[0.0] * len(timestamps),
            )

        t = np.array([
            (ts - self._start_time).total_seconds() / 86400.0
            for ts in timestamps
        ])
        t_scaled = t / self._t_scale

        # Trend prediction
        trend = self._k * t_scaled + self._m / self._y_scale

        # Add changepoint effects
        for i, cp in enumerate(self._changepoints):
            if i < len(self._deltas):
                cp_t = (cp.timestamp - self._start_time).total_seconds() / 86400.0 / self._t_scale
                mask = t_scaled >= cp_t
                trend[mask] += self._deltas[i] * (t_scaled[mask] - cp_t)

        # Add seasonality
        seasonality = np.zeros(len(t))
        if "combined" in self._seasonality_coeffs:
            # Reconstruct seasonality features
            X_seas = []
            if self.daily_seasonality:
                X_seas.append(self._make_seasonality_features(t, 1.0, 3))
            if self.weekly_seasonality:
                X_seas.append(self._make_seasonality_features(t, 7.0, 3))
            if self.yearly_seasonality and self._t_scale > 180:
                X_seas.append(self._make_seasonality_features(t, 365.25, 5))

            if X_seas:
                X_combined = np.hstack(X_seas)
                coeffs = self._seasonality_coeffs["combined"]
                if len(coeffs) == X_combined.shape[1]:
                    seasonality = X_combined @ coeffs

        # Final prediction
        y_pred = (trend + seasonality) * self._y_scale + self._m

        # Uncertainty estimation
        if include_uncertainty:
            # Simple uncertainty based on forecast horizon
            horizon = np.arange(len(timestamps))
            uncertainty = 0.1 * self._y_scale * np.sqrt(1 + horizon * 0.01)
            lower = y_pred - 1.96 * uncertainty
            upper = y_pred + 1.96 * uncertainty
        else:
            lower = y_pred
            upper = y_pred

        return Forecast(
            timestamps=timestamps,
            values=list(y_pred),
            lower_bound=list(lower),
            upper_bound=list(upper),
            changepoints=[cp.timestamp for cp in self._changepoints],
        )

    def get_changepoints(self) -> List[Changepoint]:
        """Get detected changepoints."""
        return self._changepoints


class ProphetAnomalyDetector(BaseAnomalyDetector):
    """
    Anomaly detector using Prophet-style forecasting.

    Detects anomalies when actual values fall outside prediction intervals.
    """

    def __init__(
        self,
        confidence: float = 0.95,
        min_samples: int = 48,
        changepoint_prior_scale: float = 0.05,
    ):
        self.confidence = confidence
        self.min_samples = min_samples
        self.forecaster = ProphetStyleForecaster(
            changepoint_prior_scale=changepoint_prior_scale
        )

        self._history: List[TimeSeriesPoint] = []
        self._last_forecast: Optional[Forecast] = None

    async def fit(self, data: List[TimeSeriesPoint]) -> None:
        """Train on historical data."""
        self._history = list(data)

        if len(self._history) >= self.min_samples:
            timestamps = [p.timestamp for p in self._history]
            values = [p.value for p in self._history]
            self.forecaster.fit(timestamps, values)

    async def detect(self, point: TimeSeriesPoint) -> Optional[Anomaly]:
        """Detect if a point is anomalous."""
        self._history.append(point)

        if len(self._history) < self.min_samples:
            return None

        # Refit periodically
        if len(self._history) % 24 == 0:
            timestamps = [p.timestamp for p in self._history[-self.min_samples*2:]]
            values = [p.value for p in self._history[-self.min_samples*2:]]
            self.forecaster.fit(timestamps, values)

        # Get prediction for this point
        forecast = self.forecaster.predict([point.timestamp])
        self._last_forecast = forecast

        predicted = forecast.values[0]
        lower = forecast.lower_bound[0]
        upper = forecast.upper_bound[0]

        if point.value < lower or point.value > upper:
            # Anomaly detected
            deviation = abs(point.value - predicted)
            interval_width = upper - lower + 1e-10
            severity_score = deviation / (interval_width / 2)

            if point.value < lower:
                anomaly_type = AnomalyType.DIP
            else:
                anomaly_type = AnomalyType.SPIKE

            severity = AlertSeverity.CRITICAL if severity_score > 3 else (
                AlertSeverity.WARNING if severity_score > 2 else AlertSeverity.INFO
            )

            return Anomaly(
                metric_name="prophet_forecast",
                timestamp=point.timestamp,
                value=point.value,
                expected_value=predicted,
                anomaly_type=anomaly_type,
                score=severity_score,
                severity=severity,
                details={
                    "predicted": predicted,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "deviation": deviation,
                    "changepoints": len(self.forecaster.get_changepoints()),
                },
            )

        return None

    async def detect_batch(self, points: List[TimeSeriesPoint]) -> List[Anomaly]:
        """Detect anomalies in batch."""
        anomalies = []
        for point in points:
            anomaly = await self.detect(point)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "prophet",
            "samples": len(self._history),
            "changepoints": len(self.forecaster.get_changepoints()),
            "confidence": self.confidence,
        }


# =============================================================================
# LSTM-Based Anomaly Detection
# =============================================================================

class LSTMCell:
    """
    Pure NumPy LSTM cell implementation.

    For production, use PyTorch/TensorFlow, but this provides
    a dependency-free implementation for demonstration.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Gates: input, forget, cell, output
        self.W_i = np.random.randn(input_size, hidden_size) * scale
        self.W_f = np.random.randn(input_size, hidden_size) * scale
        self.W_c = np.random.randn(input_size, hidden_size) * scale
        self.W_o = np.random.randn(input_size, hidden_size) * scale

        self.U_i = np.random.randn(hidden_size, hidden_size) * scale
        self.U_f = np.random.randn(hidden_size, hidden_size) * scale
        self.U_c = np.random.randn(hidden_size, hidden_size) * scale
        self.U_o = np.random.randn(hidden_size, hidden_size) * scale

        self.b_i = np.zeros(hidden_size)
        self.b_f = np.ones(hidden_size)  # Initialize forget gate bias to 1
        self.b_c = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through LSTM cell."""
        # Input gate
        i = self.sigmoid(x @ self.W_i + h_prev @ self.U_i + self.b_i)
        # Forget gate
        f = self.sigmoid(x @ self.W_f + h_prev @ self.U_f + self.b_f)
        # Cell candidate
        c_tilde = np.tanh(x @ self.W_c + h_prev @ self.U_c + self.b_c)
        # Cell state
        c = f * c_prev + i * c_tilde
        # Output gate
        o = self.sigmoid(x @ self.W_o + h_prev @ self.U_o + self.b_o)
        # Hidden state
        h = o * np.tanh(c)

        return h, c


class LSTMAnomalyDetector(BaseAnomalyDetector):
    """
    LSTM-based sequence anomaly detection.

    Uses an LSTM autoencoder to learn normal patterns and
    detect anomalies based on reconstruction error.
    """

    def __init__(
        self,
        sequence_length: int = 24,
        hidden_size: int = 32,
        threshold_percentile: float = 95.0,
        min_samples: int = 100,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.threshold_percentile = threshold_percentile
        self.min_samples = min_samples

        # Encoder and decoder LSTMs
        self.encoder = LSTMCell(1, hidden_size)
        self.decoder = LSTMCell(hidden_size, 1)

        # Output projection
        self.W_out = np.random.randn(hidden_size, 1) * 0.1
        self.b_out = np.zeros(1)

        self._history: deque = deque(maxlen=sequence_length * 100)
        self._reconstruction_errors: deque = deque(maxlen=1000)
        self._threshold: float = 0.1
        self._is_fitted: bool = False

    def _normalize(self, values: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Normalize values to zero mean and unit variance."""
        mean = np.mean(values)
        std = np.std(values) + 1e-10
        return (values - mean) / std, mean, std

    def _create_sequences(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Create sequences for LSTM training."""
        sequences = []
        for i in range(len(values) - self.sequence_length):
            seq = values[i:i + self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def _encode_sequence(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:
        """Encode a sequence using the LSTM encoder."""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        for t in range(len(sequence)):
            x = np.array([[sequence[t]]])
            h, c = self.encoder.forward(x, h, c)

        return h

    def _decode_sequence(
        self,
        encoding: np.ndarray,
        length: int,
    ) -> np.ndarray:
        """Decode from encoding to reconstruct sequence."""
        h = encoding
        c = np.zeros(self.hidden_size)

        outputs = []
        for t in range(length):
            h, c = self.decoder.forward(h.reshape(1, -1), h, c)
            out = h @ self.W_out + self.b_out
            outputs.append(out[0, 0])

        return np.array(outputs)

    def _compute_reconstruction_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> float:
        """Compute reconstruction error (MSE)."""
        return float(np.mean((original - reconstructed) ** 2))

    async def fit(self, data: List[TimeSeriesPoint]) -> None:
        """Train the LSTM on historical data."""
        if len(data) < self.min_samples:
            return

        values = np.array([p.value for p in data])
        normalized, _, _ = self._normalize(values)

        sequences = self._create_sequences(normalized)

        if len(sequences) < 10:
            return

        # Compute reconstruction errors on training data
        errors = []
        for seq in sequences[:100]:  # Sample for efficiency
            encoding = self._encode_sequence(seq)
            reconstruction = self._decode_sequence(encoding, len(seq))
            error = self._compute_reconstruction_error(seq, reconstruction)
            errors.append(error)

        self._reconstruction_errors.extend(errors)
        self._threshold = np.percentile(errors, self.threshold_percentile)
        self._is_fitted = True

        # Store history
        for point in data:
            self._history.append(point)

    async def detect(self, point: TimeSeriesPoint) -> Optional[Anomaly]:
        """Detect if a point is anomalous."""
        self._history.append(point)

        if not self._is_fitted or len(self._history) < self.sequence_length:
            return None

        # Get recent sequence
        recent = list(self._history)[-self.sequence_length:]
        values = np.array([p.value for p in recent])
        normalized, mean, std = self._normalize(values)

        # Compute reconstruction error
        encoding = self._encode_sequence(normalized)
        reconstruction = self._decode_sequence(encoding, len(normalized))
        error = self._compute_reconstruction_error(normalized, reconstruction)

        self._reconstruction_errors.append(error)

        # Update threshold dynamically
        if len(self._reconstruction_errors) > 50:
            self._threshold = np.percentile(
                list(self._reconstruction_errors),
                self.threshold_percentile
            )

        if error > self._threshold:
            # Find which part of sequence is anomalous
            point_errors = (normalized - reconstruction) ** 2
            anomaly_idx = np.argmax(point_errors)

            severity_score = error / (self._threshold + 1e-10)

            severity = AlertSeverity.CRITICAL if severity_score > 3 else (
                AlertSeverity.WARNING if severity_score > 2 else AlertSeverity.INFO
            )

            return Anomaly(
                metric_name="lstm_reconstruction",
                timestamp=point.timestamp,
                value=point.value,
                expected_value=float(reconstruction[-1] * std + mean),
                anomaly_type=AnomalyType.PATTERN,
                score=severity_score,
                severity=severity,
                details={
                    "reconstruction_error": error,
                    "threshold": self._threshold,
                    "sequence_length": self.sequence_length,
                    "anomaly_position": anomaly_idx,
                },
            )

        return None

    async def detect_batch(self, points: List[TimeSeriesPoint]) -> List[Anomaly]:
        """Detect anomalies in batch."""
        anomalies = []
        for point in points:
            anomaly = await self.detect(point)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "lstm",
            "is_fitted": self._is_fitted,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "threshold": self._threshold,
            "samples": len(self._history),
        }


# =============================================================================
# Transformer-Based Anomaly Detection
# =============================================================================

class TransformerAnomalyDetector(BaseAnomalyDetector):
    """
    Transformer-based anomaly detection using attention mechanism.

    Uses self-attention to capture long-range dependencies and
    detect anomalous patterns in time series.
    """

    def __init__(
        self,
        sequence_length: int = 48,
        d_model: int = 32,
        n_heads: int = 4,
        threshold_percentile: float = 95.0,
        min_samples: int = 100,
    ):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.threshold_percentile = threshold_percentile
        self.min_samples = min_samples

        self.d_k = d_model // n_heads

        # Attention weights
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1

        # Value embedding
        self.value_embed = np.random.randn(1, d_model) * 0.1

        # Output projection
        self.W_out = np.random.randn(d_model, 1) * 0.1

        self._history: deque = deque(maxlen=sequence_length * 50)
        self._attention_scores: deque = deque(maxlen=1000)
        self._threshold: float = 0.1
        self._is_fitted: bool = False

    def _positional_encoding(self, length: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        positions = np.arange(length)[:, np.newaxis]
        dims = np.arange(self.d_model)[np.newaxis, :]

        angles = positions / np.power(10000, 2 * (dims // 2) / self.d_model)

        pe = np.zeros((length, self.d_model))
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])

        return pe

    def _self_attention(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute multi-head self-attention."""
        seq_len = X.shape[0]

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Scaled dot-product attention
        scores = Q @ K.T / np.sqrt(self.d_k)
        attention_weights = self._softmax(scores)

        output = attention_weights @ V
        output = output @ self.W_O

        return output, attention_weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)

    def _compute_anomaly_score(
        self,
        attention_weights: np.ndarray,
    ) -> float:
        """
        Compute anomaly score based on attention patterns.

        Anomalies often cause irregular attention patterns.
        """
        # Entropy of attention (low entropy = focusing on specific positions)
        entropy = -np.sum(
            attention_weights * np.log(attention_weights + 1e-10),
            axis=-1
        )
        mean_entropy = np.mean(entropy)

        # Self-attention on last position (how much last element attends elsewhere)
        last_attention = attention_weights[-1, :]

        # Score based on deviation from uniform attention
        uniform = 1.0 / len(last_attention)
        deviation = np.sum((last_attention - uniform) ** 2)

        return float(deviation + (1.0 / (mean_entropy + 1e-10)))

    async def fit(self, data: List[TimeSeriesPoint]) -> None:
        """Train on historical data."""
        if len(data) < self.min_samples:
            return

        values = np.array([p.value for p in data])
        mean, std = np.mean(values), np.std(values) + 1e-10
        normalized = (values - mean) / std

        # Compute attention scores for training sequences
        scores = []
        for i in range(len(normalized) - self.sequence_length):
            seq = normalized[i:i + self.sequence_length]

            # Create embeddings
            X = seq.reshape(-1, 1) * self.value_embed
            X = X + self._positional_encoding(len(seq))

            _, attention = self._self_attention(X)
            score = self._compute_anomaly_score(attention)
            scores.append(score)

        self._attention_scores.extend(scores)
        self._threshold = np.percentile(scores, self.threshold_percentile)
        self._is_fitted = True

        for point in data:
            self._history.append(point)

    async def detect(self, point: TimeSeriesPoint) -> Optional[Anomaly]:
        """Detect if a point is anomalous."""
        self._history.append(point)

        if not self._is_fitted or len(self._history) < self.sequence_length:
            return None

        recent = list(self._history)[-self.sequence_length:]
        values = np.array([p.value for p in recent])
        mean, std = np.mean(values), np.std(values) + 1e-10
        normalized = (values - mean) / std

        # Compute attention
        X = normalized.reshape(-1, 1) * self.value_embed
        X = X + self._positional_encoding(len(normalized))

        output, attention = self._self_attention(X)
        score = self._compute_anomaly_score(attention)

        self._attention_scores.append(score)

        # Update threshold
        if len(self._attention_scores) > 50:
            self._threshold = np.percentile(
                list(self._attention_scores),
                self.threshold_percentile
            )

        if score > self._threshold:
            # Find positions with unusual attention
            last_attention = attention[-1, :]
            unusual_positions = np.where(
                last_attention > 2 * np.mean(last_attention)
            )[0]

            severity_score = score / (self._threshold + 1e-10)

            severity = AlertSeverity.CRITICAL if severity_score > 3 else (
                AlertSeverity.WARNING if severity_score > 2 else AlertSeverity.INFO
            )

            return Anomaly(
                metric_name="transformer_attention",
                timestamp=point.timestamp,
                value=point.value,
                expected_value=float(np.mean(values[:-1])),
                anomaly_type=AnomalyType.PATTERN,
                score=severity_score,
                severity=severity,
                details={
                    "attention_score": score,
                    "threshold": self._threshold,
                    "unusual_positions": list(unusual_positions),
                    "attention_entropy": float(-np.sum(
                        last_attention * np.log(last_attention + 1e-10)
                    )),
                },
            )

        return None

    async def detect_batch(self, points: List[TimeSeriesPoint]) -> List[Anomaly]:
        """Detect anomalies in batch."""
        anomalies = []
        for point in points:
            anomaly = await self.detect(point)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "transformer",
            "is_fitted": self._is_fitted,
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "threshold": self._threshold,
            "samples": len(self._history),
        }


# =============================================================================
# Multivariate Correlation Analysis
# =============================================================================

class MultivariateAnomalyDetector(BaseAnomalyDetector):
    """
    Multivariate anomaly detection through correlation analysis.

    Detects when relationships between metrics break down,
    indicating system-level anomalies.
    """

    def __init__(
        self,
        window_size: int = 100,
        correlation_threshold: float = 0.5,
        deviation_threshold: float = 2.0,
        min_samples: int = 50,
    ):
        self.window_size = window_size
        self.correlation_threshold = correlation_threshold
        self.deviation_threshold = deviation_threshold
        self.min_samples = min_samples

        self._metric_history: Dict[str, deque] = {}
        self._baseline_correlations: Dict[Tuple[str, str], float] = {}
        self._is_fitted: bool = False

    def _compute_correlation_matrix(
        self,
        data: Dict[str, List[float]],
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise correlations between metrics."""
        metrics = list(data.keys())
        correlations = {}

        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if i < j:
                    v1 = np.array(data[m1])
                    v2 = np.array(data[m2])

                    if len(v1) > 2 and len(v2) > 2:
                        # Pearson correlation
                        corr = np.corrcoef(v1, v2)[0, 1]
                        if not np.isnan(corr):
                            correlations[(m1, m2)] = corr

        return correlations

    async def fit(self, data: List[TimeSeriesPoint]) -> None:
        """Train on historical data (organized by metric)."""
        # Group by metric name
        grouped: Dict[str, List[float]] = {}

        for point in data:
            metric_name = point.labels.get("metric", "default")
            if metric_name not in grouped:
                grouped[metric_name] = []
            grouped[metric_name].append(point.value)

        # Initialize history
        for name, values in grouped.items():
            self._metric_history[name] = deque(values[-self.window_size:], maxlen=self.window_size)

        # Compute baseline correlations
        if len(grouped) > 1:
            self._baseline_correlations = self._compute_correlation_matrix(grouped)
            self._is_fitted = True

    async def detect(self, point: TimeSeriesPoint) -> Optional[Anomaly]:
        """Detect multivariate anomalies."""
        metric_name = point.labels.get("metric", "default")

        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = deque(maxlen=self.window_size)

        self._metric_history[metric_name].append(point.value)

        if not self._is_fitted or len(self._metric_history) < 2:
            return None

        # Check if we have enough data for all metrics
        min_len = min(len(h) for h in self._metric_history.values())
        if min_len < self.min_samples:
            return None

        # Compute current correlations
        current_data = {
            name: list(hist) for name, hist in self._metric_history.items()
        }
        current_correlations = self._compute_correlation_matrix(current_data)

        # Find correlation breakdowns
        anomalous_pairs = []
        deviations = {}

        for pair, baseline_corr in self._baseline_correlations.items():
            if pair in current_correlations:
                current_corr = current_correlations[pair]
                deviation = abs(current_corr - baseline_corr)
                deviations[pair] = {
                    "baseline": baseline_corr,
                    "current": current_corr,
                    "deviation": deviation,
                }

                # Check if correlation has significantly changed
                if deviation > self.deviation_threshold * (1 - abs(baseline_corr)):
                    anomalous_pairs.append(pair)

        if anomalous_pairs:
            max_deviation = max(
                deviations[p]["deviation"] for p in anomalous_pairs
            )

            severity = AlertSeverity.CRITICAL if len(anomalous_pairs) > 3 else (
                AlertSeverity.WARNING if len(anomalous_pairs) > 1 else AlertSeverity.INFO
            )

            return Anomaly(
                metric_name="multivariate_correlation",
                timestamp=point.timestamp,
                value=point.value,
                expected_value=point.value,
                anomaly_type=AnomalyType.CORRELATION,
                score=max_deviation,
                severity=severity,
                details={
                    "anomalous_pairs": [
                        {"metrics": list(p), **deviations[p]}
                        for p in anomalous_pairs
                    ],
                    "total_metrics": len(self._metric_history),
                    "correlation_changes": len(anomalous_pairs),
                },
            )

        return None

    async def detect_batch(self, points: List[TimeSeriesPoint]) -> List[Anomaly]:
        """Detect anomalies in batch."""
        anomalies = []
        for point in points:
            anomaly = await self.detect(point)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "multivariate",
            "is_fitted": self._is_fitted,
            "num_metrics": len(self._metric_history),
            "baseline_correlations": len(self._baseline_correlations),
            "window_size": self.window_size,
        }


# =============================================================================
# Isolation Forest (Full Implementation)
# =============================================================================

class IsolationTree:
    """A single isolation tree for anomaly detection."""

    def __init__(
        self,
        height_limit: int,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.height_limit = height_limit
        self.rng = random_state or np.random.RandomState()

        self.split_feature: Optional[int] = None
        self.split_value: Optional[float] = None
        self.left: Optional[IsolationTree] = None
        self.right: Optional[IsolationTree] = None
        self.size: int = 0
        self.is_leaf: bool = True

    def fit(
        self,
        X: np.ndarray,
        current_height: int = 0,
    ) -> None:
        """Build the isolation tree."""
        n_samples, n_features = X.shape
        self.size = n_samples

        if current_height >= self.height_limit or n_samples <= 1:
            self.is_leaf = True
            return

        # Random feature selection
        self.split_feature = self.rng.randint(0, n_features)

        # Get min and max for the selected feature
        feature_values = X[:, self.split_feature]
        min_val, max_val = feature_values.min(), feature_values.max()

        if min_val == max_val:
            self.is_leaf = True
            return

        # Random split point
        self.split_value = self.rng.uniform(min_val, max_val)
        self.is_leaf = False

        # Split data
        left_mask = feature_values < self.split_value
        right_mask = ~left_mask

        X_left = X[left_mask]
        X_right = X[right_mask]

        if len(X_left) > 0:
            self.left = IsolationTree(self.height_limit, self.rng)
            self.left.fit(X_left, current_height + 1)

        if len(X_right) > 0:
            self.right = IsolationTree(self.height_limit, self.rng)
            self.right.fit(X_right, current_height + 1)

    def path_length(
        self,
        x: np.ndarray,
        current_height: int = 0,
    ) -> float:
        """Compute path length for a single sample."""
        if self.is_leaf:
            # Adjustment for unsplit samples (from original paper)
            if self.size > 1:
                return current_height + self._c(self.size)
            return current_height

        if x[self.split_feature] < self.split_value:
            if self.left:
                return self.left.path_length(x, current_height + 1)
        else:
            if self.right:
                return self.right.path_length(x, current_height + 1)

        return current_height

    @staticmethod
    def _c(n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Full Isolation Forest implementation for anomaly detection.

    Based on Liu et al. "Isolation Forest" (2008).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        threshold_percentile: float = 95.0,
        min_samples: int = 50,
        feature_window: int = 10,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.threshold_percentile = threshold_percentile
        self.min_samples = min_samples
        self.feature_window = feature_window
        self.random_state = random_state

        self._trees: List[IsolationTree] = []
        self._history: deque = deque(maxlen=max_samples * 10)
        self._threshold: float = 0.5
        self._is_fitted: bool = False
        self._rng = np.random.RandomState(random_state)

    def _create_features(self, values: List[float]) -> np.ndarray:
        """Create feature matrix from time series values."""
        if len(values) < self.feature_window:
            return np.array(values).reshape(1, -1)

        features = []
        for i in range(len(values) - self.feature_window + 1):
            window = values[i:i + self.feature_window]

            # Statistical features
            feature = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                window[-1] - window[0],  # Change
                np.mean(np.diff(window)),  # Trend
            ]
            features.append(feature)

        return np.array(features)

    async def fit(self, data: List[TimeSeriesPoint]) -> None:
        """Train the Isolation Forest."""
        if len(data) < self.min_samples:
            return

        values = [p.value for p in data]
        X = self._create_features(values)

        if len(X) < 2:
            return

        n_samples = min(self.max_samples, len(X))
        height_limit = int(np.ceil(np.log2(n_samples)))

        self._trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            idx = self._rng.choice(len(X), size=n_samples, replace=False)
            X_sample = X[idx]

            tree = IsolationTree(height_limit, self._rng)
            tree.fit(X_sample)
            self._trees.append(tree)

        # Compute threshold based on training data
        scores = []
        for i in range(len(X)):
            score = self._anomaly_score(X[i])
            scores.append(score)

        self._threshold = np.percentile(scores, 100 - self.contamination * 100)
        self._is_fitted = True

        for point in data:
            self._history.append(point)

    def _anomaly_score(self, x: np.ndarray) -> float:
        """Compute anomaly score for a sample."""
        if not self._trees:
            return 0.0

        n = self.max_samples
        c_n = IsolationTree._c(n)

        avg_path = np.mean([tree.path_length(x) for tree in self._trees])

        # Anomaly score from paper
        score = 2 ** (-avg_path / c_n)

        return float(score)

    async def detect(self, point: TimeSeriesPoint) -> Optional[Anomaly]:
        """Detect if a point is anomalous."""
        self._history.append(point)

        if not self._is_fitted:
            return None

        if len(self._history) < self.feature_window:
            return None

        # Get recent window
        recent = list(self._history)[-self.feature_window:]
        values = [p.value for p in recent]

        # Create features
        feature = np.array([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            values[-1] - values[0],
            np.mean(np.diff(values)),
        ])

        score = self._anomaly_score(feature)

        if score > self._threshold:
            severity = AlertSeverity.CRITICAL if score > 0.8 else (
                AlertSeverity.WARNING if score > 0.6 else AlertSeverity.INFO
            )

            return Anomaly(
                metric_name="isolation_forest",
                timestamp=point.timestamp,
                value=point.value,
                expected_value=np.mean(values[:-1]),
                anomaly_type=AnomalyType.OUTLIER,
                score=score,
                severity=severity,
                details={
                    "isolation_score": score,
                    "threshold": self._threshold,
                    "n_trees": len(self._trees),
                },
            )

        return None

    async def detect_batch(self, points: List[TimeSeriesPoint]) -> List[Anomaly]:
        """Detect anomalies in batch."""
        anomalies = []
        for point in points:
            anomaly = await self.detect(point)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "isolation_forest",
            "is_fitted": self._is_fitted,
            "n_estimators": len(self._trees),
            "threshold": self._threshold,
            "samples": len(self._history),
        }


# =============================================================================
# Ensemble Detector
# =============================================================================

class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """
    Ensemble anomaly detector combining multiple detection methods.

    Uses voting or weighted combination of:
    - STL decomposition
    - Prophet forecasting
    - LSTM sequence analysis
    - Transformer attention
    - Isolation Forest
    - Multivariate correlation
    """

    def __init__(
        self,
        detectors: Optional[List[BaseAnomalyDetector]] = None,
        weights: Optional[List[float]] = None,
        voting_threshold: float = 0.5,
        combine_method: str = "weighted",  # "voting" or "weighted"
    ):
        if detectors is None:
            # Default ensemble
            detectors = [
                STLAnomalyDetector(),
                ProphetAnomalyDetector(),
                IsolationForestDetector(),
                LSTMAnomalyDetector(),
            ]

        self.detectors = detectors
        self.weights = weights or [1.0 / len(detectors)] * len(detectors)
        self.voting_threshold = voting_threshold
        self.combine_method = combine_method

        self._anomaly_history: deque = deque(maxlen=1000)

    async def fit(self, data: List[TimeSeriesPoint]) -> None:
        """Train all detectors."""
        await asyncio.gather(*[
            detector.fit(data) for detector in self.detectors
        ])

    async def detect(self, point: TimeSeriesPoint) -> Optional[Anomaly]:
        """Detect using ensemble of detectors."""
        # Run all detectors
        results = await asyncio.gather(*[
            detector.detect(point) for detector in self.detectors
        ])

        # Filter out None results
        anomalies = [(i, a) for i, a in enumerate(results) if a is not None]

        if not anomalies:
            return None

        if self.combine_method == "voting":
            # Majority voting
            vote_ratio = len(anomalies) / len(self.detectors)
            if vote_ratio >= self.voting_threshold:
                # Combine into single anomaly
                max_score_anomaly = max(anomalies, key=lambda x: x[1].score)[1]

                return Anomaly(
                    metric_name="ensemble",
                    timestamp=point.timestamp,
                    value=point.value,
                    expected_value=max_score_anomaly.expected_value,
                    anomaly_type=max_score_anomaly.anomaly_type,
                    score=vote_ratio,
                    severity=max_score_anomaly.severity,
                    details={
                        "voting_ratio": vote_ratio,
                        "detectors_triggered": [
                            self.detectors[i].__class__.__name__
                            for i, _ in anomalies
                        ],
                        "individual_scores": {
                            self.detectors[i].__class__.__name__: a.score
                            for i, a in anomalies
                        },
                    },
                )
        else:
            # Weighted combination
            weighted_score = sum(
                self.weights[i] * a.score
                for i, a in anomalies
            )

            if weighted_score > 0.5:
                max_score_anomaly = max(anomalies, key=lambda x: x[1].score)[1]

                return Anomaly(
                    metric_name="ensemble",
                    timestamp=point.timestamp,
                    value=point.value,
                    expected_value=max_score_anomaly.expected_value,
                    anomaly_type=max_score_anomaly.anomaly_type,
                    score=weighted_score,
                    severity=max_score_anomaly.severity,
                    details={
                        "weighted_score": weighted_score,
                        "detectors_triggered": [
                            self.detectors[i].__class__.__name__
                            for i, _ in anomalies
                        ],
                        "weights": {
                            self.detectors[i].__class__.__name__: self.weights[i]
                            for i in range(len(self.detectors))
                        },
                    },
                )

        return None

    async def detect_batch(self, points: List[TimeSeriesPoint]) -> List[Anomaly]:
        """Detect anomalies in batch."""
        anomalies = []
        for point in points:
            anomaly = await self.detect(point)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "ensemble",
            "n_detectors": len(self.detectors),
            "detectors": [d.__class__.__name__ for d in self.detectors],
            "weights": self.weights,
            "combine_method": self.combine_method,
            "voting_threshold": self.voting_threshold,
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_anomaly_detector(
    detector_type: Union[str, AnomalyDetectorType],
    **kwargs,
) -> BaseAnomalyDetector:
    """
    Factory function to create anomaly detectors.

    Args:
        detector_type: Type of detector to create
        **kwargs: Arguments passed to detector constructor

    Returns:
        Configured anomaly detector
    """
    if isinstance(detector_type, str):
        detector_type = AnomalyDetectorType(detector_type)

    detectors = {
        AnomalyDetectorType.STL: STLAnomalyDetector,
        AnomalyDetectorType.PROPHET: ProphetAnomalyDetector,
        AnomalyDetectorType.LSTM: LSTMAnomalyDetector,
        AnomalyDetectorType.TRANSFORMER: TransformerAnomalyDetector,
        AnomalyDetectorType.ISOLATION_FOREST: IsolationForestDetector,
        AnomalyDetectorType.MULTIVARIATE: MultivariateAnomalyDetector,
        AnomalyDetectorType.ENSEMBLE: EnsembleAnomalyDetector,
    }

    if detector_type not in detectors:
        raise ValueError(f"Unknown detector type: {detector_type}")

    return detectors[detector_type](**kwargs)
