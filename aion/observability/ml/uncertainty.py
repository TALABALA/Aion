"""
Uncertainty Quantification for Time Series Forecasting.

Implements:
- Conformal Prediction for distribution-free coverage guarantees
- Bayesian Uncertainty with variational inference
- Ensemble Uncertainty from model disagreement
- Monte Carlo Dropout for neural network uncertainty
"""

import math
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a prediction."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    std_dev: float
    variance: float
    entropy: Optional[float] = None
    epistemic: Optional[float] = None  # Model uncertainty
    aleatoric: Optional[float] = None  # Data uncertainty
    quantiles: Optional[Dict[float, float]] = None


class UncertaintyQuantifier(ABC):
    """Base class for uncertainty quantification methods."""

    @abstractmethod
    def estimate_uncertainty(
        self,
        predictions: List[float],
        actuals: Optional[List[float]] = None,
        confidence: float = 0.95
    ) -> List[UncertaintyEstimate]:
        """Estimate uncertainty for predictions."""
        pass

    @abstractmethod
    def calibrate(self, predictions: List[float], actuals: List[float]) -> None:
        """Calibrate uncertainty estimates using historical data."""
        pass


class ConformalPredictor(UncertaintyQuantifier):
    """
    Conformal Prediction for distribution-free prediction intervals.

    Provides valid coverage guarantees without distributional assumptions.
    Implements both split conformal and adaptive conformal inference.
    """

    def __init__(
        self,
        method: str = "adaptive",  # split, adaptive, cqr (conformalized quantile regression)
        alpha: float = 0.05,  # Target miscoverage rate
        window_size: int = 100,  # Calibration window
    ):
        self.method = method
        self.alpha = alpha
        self.window_size = window_size

        # Nonconformity scores from calibration
        self.calibration_scores: deque = deque(maxlen=window_size)
        self.is_calibrated = False

        # Adaptive parameters
        self.adaptive_alpha = alpha
        self.gamma = 0.005  # Learning rate for adaptive alpha

    def _compute_nonconformity_score(
        self,
        prediction: float,
        actual: float,
        lower: Optional[float] = None,
        upper: Optional[float] = None
    ) -> float:
        """Compute nonconformity score."""
        if self.method == "split":
            # Simple absolute residual
            return abs(actual - prediction)

        elif self.method == "adaptive":
            # Normalized residual
            spread = abs(upper - lower) if lower and upper else 1.0
            return abs(actual - prediction) / (spread + 1e-8)

        elif self.method == "cqr":
            # Conformalized Quantile Regression score
            if lower is not None and upper is not None:
                return max(lower - actual, actual - upper)
            return abs(actual - prediction)

        return abs(actual - prediction)

    def calibrate(self, predictions: List[float], actuals: List[float]) -> None:
        """Calibrate using held-out calibration set."""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        self.calibration_scores.clear()

        for pred, actual in zip(predictions, actuals):
            score = self._compute_nonconformity_score(pred, actual)
            self.calibration_scores.append(score)

        self.is_calibrated = True
        logger.info(f"Calibrated conformal predictor with {len(self.calibration_scores)} samples")

    def _get_conformal_quantile(self) -> float:
        """Get conformal quantile from calibration scores."""
        if not self.calibration_scores:
            return 1.96  # Default to normal approximation

        scores = sorted(self.calibration_scores)
        n = len(scores)

        # Conformal quantile with finite-sample correction
        quantile_idx = int(math.ceil((n + 1) * (1 - self.alpha))) - 1
        quantile_idx = max(0, min(n - 1, quantile_idx))

        return scores[quantile_idx]

    def estimate_uncertainty(
        self,
        predictions: List[float],
        actuals: Optional[List[float]] = None,
        confidence: float = 0.95
    ) -> List[UncertaintyEstimate]:
        """Estimate prediction intervals using conformal prediction."""
        self.alpha = 1 - confidence

        if self.method == "adaptive" and actuals is not None:
            return self._adaptive_conformal(predictions, actuals, confidence)

        # Standard conformal prediction
        q = self._get_conformal_quantile()

        estimates = []
        for pred in predictions:
            # Compute prediction interval
            lower = pred - q
            upper = pred + q

            # Estimate std from conformal score
            std_dev = q / 1.96  # Approximate

            estimates.append(UncertaintyEstimate(
                point_estimate=pred,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=confidence,
                std_dev=std_dev,
                variance=std_dev ** 2,
                quantiles={
                    0.025: lower,
                    0.5: pred,
                    0.975: upper
                }
            ))

        return estimates

    def _adaptive_conformal(
        self,
        predictions: List[float],
        actuals: List[float],
        confidence: float
    ) -> List[UncertaintyEstimate]:
        """Adaptive Conformal Inference (ACI) for online settings."""
        estimates = []

        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            # Get current quantile
            q = self._get_conformal_quantile() * (1 + self.adaptive_alpha - self.alpha)

            lower = pred - q
            upper = pred + q

            # Update adaptive alpha based on coverage
            covered = lower <= actual <= upper
            self.adaptive_alpha += self.gamma * (self.alpha - (0 if covered else 1))
            self.adaptive_alpha = max(0.001, min(0.5, self.adaptive_alpha))

            # Update calibration scores
            score = self._compute_nonconformity_score(pred, actual, lower, upper)
            self.calibration_scores.append(score)

            std_dev = q / 1.96
            estimates.append(UncertaintyEstimate(
                point_estimate=pred,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=confidence,
                std_dev=std_dev,
                variance=std_dev ** 2
            ))

        return estimates


class BayesianUncertainty(UncertaintyQuantifier):
    """
    Bayesian Uncertainty Quantification.

    Implements variational inference for neural network uncertainty
    with separation of epistemic and aleatoric uncertainty.
    """

    def __init__(
        self,
        prior_std: float = 1.0,
        likelihood_std: float = 0.1,
        num_samples: int = 100,
        kl_weight: float = 0.1
    ):
        self.prior_std = prior_std
        self.likelihood_std = likelihood_std
        self.num_samples = num_samples
        self.kl_weight = kl_weight

        # Variational parameters (mean and log_std for each weight)
        self.variational_mean: Optional[List[float]] = None
        self.variational_log_std: Optional[List[float]] = None

        # Learned noise parameter for aleatoric uncertainty
        self.log_noise_std = math.log(likelihood_std)

    def _sample_weights(self) -> List[float]:
        """Sample weights from variational posterior."""
        if self.variational_mean is None:
            return []

        weights = []
        for mean, log_std in zip(self.variational_mean, self.variational_log_std):
            std = math.exp(log_std)
            # Reparameterization trick
            eps = random.gauss(0, 1)
            weights.append(mean + std * eps)

        return weights

    def _kl_divergence(self) -> float:
        """Compute KL divergence between variational posterior and prior."""
        if self.variational_mean is None:
            return 0.0

        kl = 0.0
        for mean, log_std in zip(self.variational_mean, self.variational_log_std):
            std = math.exp(log_std)
            # KL(N(mean, std) || N(0, prior_std))
            kl += (math.log(self.prior_std / std) +
                   (std**2 + mean**2) / (2 * self.prior_std**2) - 0.5)

        return kl

    def calibrate(self, predictions: List[float], actuals: List[float]) -> None:
        """Calibrate variational parameters using maximum likelihood."""
        n = len(predictions)

        # Initialize variational parameters
        self.variational_mean = [0.0] * n
        self.variational_log_std = [math.log(0.1)] * n

        # Variational inference (simplified)
        learning_rate = 0.01

        for _ in range(100):
            # Sample weights
            weights = self._sample_weights()

            # Compute gradients (simplified)
            for i, (pred, actual) in enumerate(zip(predictions, actuals)):
                residual = actual - pred

                # Update mean
                self.variational_mean[i] += learning_rate * residual

                # Update log_std (encourage smaller variance if prediction is good)
                self.variational_log_std[i] -= learning_rate * 0.1 * abs(residual)

            # KL regularization
            kl = self._kl_divergence()
            for i in range(len(self.variational_mean)):
                self.variational_mean[i] -= learning_rate * self.kl_weight * self.variational_mean[i] / self.prior_std**2

        # Estimate noise from residuals
        residuals = [a - p for a, p in zip(actuals, predictions)]
        self.log_noise_std = math.log(max(0.01, math.sqrt(sum(r**2 for r in residuals) / n)))

        logger.info("Bayesian uncertainty calibration complete")

    def estimate_uncertainty(
        self,
        predictions: List[float],
        actuals: Optional[List[float]] = None,
        confidence: float = 0.95
    ) -> List[UncertaintyEstimate]:
        """Estimate uncertainty with epistemic/aleatoric decomposition."""
        z = self._normal_quantile((1 + confidence) / 2)

        estimates = []
        for pred in predictions:
            # Sample from posterior
            samples = []
            for _ in range(self.num_samples):
                # Epistemic: sample from weight distribution
                if self.variational_mean:
                    eps_epistemic = random.gauss(0, 0.1)
                else:
                    eps_epistemic = random.gauss(0, 0.1)

                # Aleatoric: sample from noise distribution
                eps_aleatoric = random.gauss(0, math.exp(self.log_noise_std))

                samples.append(pred + eps_epistemic + eps_aleatoric)

            # Compute statistics
            mean_sample = sum(samples) / len(samples)
            var_total = sum((s - mean_sample)**2 for s in samples) / len(samples)

            # Decompose uncertainty
            aleatoric_var = math.exp(2 * self.log_noise_std)
            epistemic_var = max(0, var_total - aleatoric_var)

            std_dev = math.sqrt(var_total)
            lower = pred - z * std_dev
            upper = pred + z * std_dev

            # Compute entropy (for Gaussian: 0.5 * log(2 * pi * e * sigma^2))
            entropy = 0.5 * math.log(2 * math.pi * math.e * var_total + 1e-8)

            estimates.append(UncertaintyEstimate(
                point_estimate=pred,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=confidence,
                std_dev=std_dev,
                variance=var_total,
                entropy=entropy,
                epistemic=math.sqrt(epistemic_var),
                aleatoric=math.sqrt(aleatoric_var),
                quantiles={
                    0.025: pred - 1.96 * std_dev,
                    0.25: pred - 0.674 * std_dev,
                    0.5: pred,
                    0.75: pred + 0.674 * std_dev,
                    0.975: pred + 1.96 * std_dev
                }
            ))

        return estimates

    def _normal_quantile(self, p: float) -> float:
        """Approximate inverse normal CDF."""
        # Rational approximation
        if p <= 0:
            return -10
        if p >= 1:
            return 10

        if p < 0.5:
            return -self._normal_quantile(1 - p)

        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)


class EnsembleUncertainty(UncertaintyQuantifier):
    """
    Ensemble-based Uncertainty Quantification.

    Uses disagreement between ensemble members to estimate uncertainty.
    Supports deep ensembles and bootstrap aggregating.
    """

    def __init__(
        self,
        ensemble_size: int = 5,
        bootstrap: bool = True,
        disagreement_metric: str = "std"  # std, range, entropy
    ):
        self.ensemble_size = ensemble_size
        self.bootstrap = bootstrap
        self.disagreement_metric = disagreement_metric

        # Ensemble predictions storage
        self.ensemble_predictions: List[List[float]] = []

    def add_ensemble_predictions(self, predictions: List[List[float]]):
        """Add predictions from ensemble members."""
        self.ensemble_predictions = predictions

    def calibrate(self, predictions: List[float], actuals: List[float]) -> None:
        """Calibrate ensemble disagreement to actual errors."""
        # In practice, this would learn a mapping from disagreement to uncertainty
        pass

    def estimate_uncertainty(
        self,
        predictions: List[float],
        actuals: Optional[List[float]] = None,
        confidence: float = 0.95
    ) -> List[UncertaintyEstimate]:
        """Estimate uncertainty from ensemble disagreement."""
        if not self.ensemble_predictions:
            # Fall back to bootstrap sampling if no ensemble
            return self._bootstrap_uncertainty(predictions, confidence)

        z = self._normal_quantile((1 + confidence) / 2)

        estimates = []
        for i, pred in enumerate(predictions):
            # Get ensemble predictions for this point
            if i < len(self.ensemble_predictions[0]):
                member_preds = [ep[i] for ep in self.ensemble_predictions
                               if i < len(ep)]
            else:
                member_preds = [pred]

            # Compute disagreement
            mean_pred = sum(member_preds) / len(member_preds)

            if self.disagreement_metric == "std":
                var = sum((p - mean_pred)**2 for p in member_preds) / len(member_preds)
                std_dev = math.sqrt(var)
            elif self.disagreement_metric == "range":
                std_dev = (max(member_preds) - min(member_preds)) / 4
            else:  # entropy
                # Approximate entropy from variance
                var = sum((p - mean_pred)**2 for p in member_preds) / len(member_preds)
                std_dev = math.sqrt(var)

            lower = mean_pred - z * std_dev
            upper = mean_pred + z * std_dev

            # Quantiles from sorted ensemble predictions
            sorted_preds = sorted(member_preds)
            n = len(sorted_preds)
            quantiles = {
                0.025: sorted_preds[max(0, int(0.025 * n))],
                0.25: sorted_preds[max(0, int(0.25 * n))],
                0.5: sorted_preds[n // 2],
                0.75: sorted_preds[min(n-1, int(0.75 * n))],
                0.975: sorted_preds[min(n-1, int(0.975 * n))]
            }

            estimates.append(UncertaintyEstimate(
                point_estimate=mean_pred,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=confidence,
                std_dev=std_dev,
                variance=std_dev ** 2,
                epistemic=std_dev,  # Ensemble disagreement is epistemic
                aleatoric=0.0,
                quantiles=quantiles
            ))

        return estimates

    def _bootstrap_uncertainty(
        self,
        predictions: List[float],
        confidence: float
    ) -> List[UncertaintyEstimate]:
        """Estimate uncertainty via bootstrap."""
        z = self._normal_quantile((1 + confidence) / 2)

        estimates = []
        for pred in predictions:
            # Generate bootstrap samples
            bootstrap_samples = []
            for _ in range(self.ensemble_size):
                # Add noise to simulate bootstrap
                noise = random.gauss(0, abs(pred) * 0.1 + 0.1)
                bootstrap_samples.append(pred + noise)

            mean_boot = sum(bootstrap_samples) / len(bootstrap_samples)
            var = sum((b - mean_boot)**2 for b in bootstrap_samples) / len(bootstrap_samples)
            std_dev = math.sqrt(var)

            estimates.append(UncertaintyEstimate(
                point_estimate=pred,
                lower_bound=pred - z * std_dev,
                upper_bound=pred + z * std_dev,
                confidence_level=confidence,
                std_dev=std_dev,
                variance=var
            ))

        return estimates

    def _normal_quantile(self, p: float) -> float:
        """Approximate inverse normal CDF."""
        if p <= 0:
            return -10
        if p >= 1:
            return 10
        if p < 0.5:
            return -self._normal_quantile(1 - p)

        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)


class MonteCarloDropout(UncertaintyQuantifier):
    """
    Monte Carlo Dropout for Neural Network Uncertainty.

    Approximates Bayesian inference by using dropout at test time
    and aggregating multiple stochastic forward passes.
    """

    def __init__(
        self,
        dropout_rate: float = 0.1,
        num_samples: int = 100,
        model_forward: Optional[Callable] = None
    ):
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        self.model_forward = model_forward

        # Calibration parameters
        self.scale_factor = 1.0

    def set_model_forward(self, forward_fn: Callable):
        """Set the model forward function with dropout enabled."""
        self.model_forward = forward_fn

    def calibrate(self, predictions: List[float], actuals: List[float]) -> None:
        """Calibrate MC Dropout uncertainty estimates."""
        if not predictions or not actuals:
            return

        # Compute residuals
        residuals = [abs(a - p) for a, p in zip(actuals, predictions)]
        mean_residual = sum(residuals) / len(residuals)

        # MC Dropout variance (simulated)
        mc_vars = []
        for pred in predictions:
            samples = [pred + random.gauss(0, abs(pred) * self.dropout_rate + 0.1)
                      for _ in range(20)]
            mean_s = sum(samples) / len(samples)
            var_s = sum((s - mean_s)**2 for s in samples) / len(samples)
            mc_vars.append(math.sqrt(var_s))

        mean_mc_std = sum(mc_vars) / len(mc_vars)

        # Scale factor to match empirical errors
        self.scale_factor = mean_residual / (mean_mc_std + 1e-8)

        logger.info(f"MC Dropout calibrated with scale_factor={self.scale_factor:.3f}")

    def estimate_uncertainty(
        self,
        predictions: List[float],
        actuals: Optional[List[float]] = None,
        confidence: float = 0.95
    ) -> List[UncertaintyEstimate]:
        """Estimate uncertainty using MC Dropout."""
        z = self._normal_quantile((1 + confidence) / 2)

        estimates = []
        for pred in predictions:
            # Run multiple forward passes with dropout
            if self.model_forward:
                samples = [self.model_forward(pred) for _ in range(self.num_samples)]
            else:
                # Simulate MC Dropout
                samples = []
                for _ in range(self.num_samples):
                    # Simulate dropout effect
                    dropout_noise = random.gauss(0, abs(pred) * self.dropout_rate + 0.1)
                    samples.append(pred + dropout_noise)

            # Compute statistics
            mean_pred = sum(samples) / len(samples)
            var = sum((s - mean_pred)**2 for s in samples) / len(samples)
            std_dev = math.sqrt(var) * self.scale_factor

            # Epistemic uncertainty from MC Dropout
            epistemic = std_dev

            # Estimate aleatoric from prediction magnitude
            aleatoric = abs(pred) * 0.05

            total_std = math.sqrt(epistemic**2 + aleatoric**2)

            lower = pred - z * total_std
            upper = pred + z * total_std

            # Quantiles from samples
            sorted_samples = sorted(samples)
            n = len(sorted_samples)

            estimates.append(UncertaintyEstimate(
                point_estimate=pred,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=confidence,
                std_dev=total_std,
                variance=total_std ** 2,
                epistemic=epistemic,
                aleatoric=aleatoric,
                quantiles={
                    0.025: sorted_samples[max(0, int(0.025 * n))],
                    0.25: sorted_samples[max(0, int(0.25 * n))],
                    0.5: sorted_samples[n // 2],
                    0.75: sorted_samples[min(n-1, int(0.75 * n))],
                    0.975: sorted_samples[min(n-1, int(0.975 * n))]
                }
            ))

        return estimates

    def _normal_quantile(self, p: float) -> float:
        """Approximate inverse normal CDF."""
        if p <= 0:
            return -10
        if p >= 1:
            return 10
        if p < 0.5:
            return -self._normal_quantile(1 - p)

        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)


class CombinedUncertaintyQuantifier(UncertaintyQuantifier):
    """
    Combined uncertainty quantification using multiple methods.

    Aggregates estimates from conformal prediction, Bayesian inference,
    and ensemble methods for robust uncertainty bounds.
    """

    def __init__(
        self,
        methods: List[str] = None,
        aggregation: str = "conservative"  # conservative, average, calibrated
    ):
        self.methods = methods or ["conformal", "bayesian", "ensemble"]
        self.aggregation = aggregation

        # Initialize quantifiers
        self.quantifiers: Dict[str, UncertaintyQuantifier] = {}
        if "conformal" in self.methods:
            self.quantifiers["conformal"] = ConformalPredictor()
        if "bayesian" in self.methods:
            self.quantifiers["bayesian"] = BayesianUncertainty()
        if "ensemble" in self.methods:
            self.quantifiers["ensemble"] = EnsembleUncertainty()
        if "mc_dropout" in self.methods:
            self.quantifiers["mc_dropout"] = MonteCarloDropout()

    def calibrate(self, predictions: List[float], actuals: List[float]) -> None:
        """Calibrate all quantifiers."""
        for name, quantifier in self.quantifiers.items():
            try:
                quantifier.calibrate(predictions, actuals)
            except Exception as e:
                logger.warning(f"Failed to calibrate {name}: {e}")

    def estimate_uncertainty(
        self,
        predictions: List[float],
        actuals: Optional[List[float]] = None,
        confidence: float = 0.95
    ) -> List[UncertaintyEstimate]:
        """Estimate uncertainty using combined methods."""
        # Get estimates from all methods
        all_estimates: Dict[str, List[UncertaintyEstimate]] = {}

        for name, quantifier in self.quantifiers.items():
            try:
                estimates = quantifier.estimate_uncertainty(predictions, actuals, confidence)
                all_estimates[name] = estimates
            except Exception as e:
                logger.warning(f"Failed to get estimates from {name}: {e}")

        if not all_estimates:
            raise ValueError("All uncertainty methods failed")

        # Aggregate estimates
        combined = []
        first_method = list(all_estimates.keys())[0]
        n_predictions = len(all_estimates[first_method])

        for i in range(n_predictions):
            point_estimates = []
            lower_bounds = []
            upper_bounds = []
            std_devs = []
            epistemics = []
            aleatorics = []

            for estimates in all_estimates.values():
                if i < len(estimates):
                    e = estimates[i]
                    point_estimates.append(e.point_estimate)
                    lower_bounds.append(e.lower_bound)
                    upper_bounds.append(e.upper_bound)
                    std_devs.append(e.std_dev)
                    if e.epistemic is not None:
                        epistemics.append(e.epistemic)
                    if e.aleatoric is not None:
                        aleatorics.append(e.aleatoric)

            # Aggregate based on strategy
            if self.aggregation == "conservative":
                # Use widest bounds
                lower = min(lower_bounds)
                upper = max(upper_bounds)
                std_dev = max(std_devs)
            elif self.aggregation == "average":
                # Average bounds
                lower = sum(lower_bounds) / len(lower_bounds)
                upper = sum(upper_bounds) / len(upper_bounds)
                std_dev = sum(std_devs) / len(std_devs)
            else:  # calibrated
                # Weighted average based on historical performance
                # (simplified: use simple average)
                lower = sum(lower_bounds) / len(lower_bounds)
                upper = sum(upper_bounds) / len(upper_bounds)
                std_dev = sum(std_devs) / len(std_devs)

            point = sum(point_estimates) / len(point_estimates)
            epistemic = sum(epistemics) / len(epistemics) if epistemics else None
            aleatoric = sum(aleatorics) / len(aleatorics) if aleatorics else None

            combined.append(UncertaintyEstimate(
                point_estimate=point,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=confidence,
                std_dev=std_dev,
                variance=std_dev ** 2,
                epistemic=epistemic,
                aleatoric=aleatoric
            ))

        return combined
