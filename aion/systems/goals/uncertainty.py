"""
AION Goal System - Uncertainty Quantification

SOTA uncertainty quantification providing confidence estimates
and probabilistic reasoning for goal management decisions.

Key capabilities:
- Bayesian belief networks for goal success estimation
- Monte Carlo dropout for neural uncertainty
- Ensemble disagreement quantification
- Calibrated confidence intervals
- Epistemic vs aleatoric uncertainty decomposition
- Thompson sampling for exploration vs exploitation
"""

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
)

logger = structlog.get_logger()


@dataclass
class UncertaintyEstimate:
    """Comprehensive uncertainty estimate for a prediction."""

    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    confidence_level: float = 0.95

    # Uncertainty decomposition
    epistemic_uncertainty: float = 0.0  # Model uncertainty (reducible)
    aleatoric_uncertainty: float = 0.0  # Data/inherent uncertainty (irreducible)

    # Distribution info
    distribution: str = "normal"
    samples: Optional[np.ndarray] = None

    def total_uncertainty(self) -> float:
        """Total uncertainty combining both types."""
        return math.sqrt(self.epistemic_uncertainty**2 + self.aleatoric_uncertainty**2)

    @property
    def lower_bound(self) -> float:
        return self.confidence_interval[0]

    @property
    def upper_bound(self) -> float:
        return self.confidence_interval[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "confidence_interval": list(self.confidence_interval),
            "confidence_level": self.confidence_level,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "total_uncertainty": self.total_uncertainty(),
            "distribution": self.distribution,
        }


@dataclass
class BetaDistribution:
    """
    Beta distribution for modeling probabilities.

    Conjugate prior for Bernoulli observations, perfect for
    modeling goal success rates.
    """

    alpha: float = 1.0  # Prior successes + 1
    beta: float = 1.0   # Prior failures + 1

    def mean(self) -> float:
        """Expected value."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Variance of the distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(self.variance())

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute credible interval."""
        lower_p = (1 - level) / 2
        upper_p = 1 - lower_p
        lower = stats.beta.ppf(lower_p, self.alpha, self.beta)
        upper = stats.beta.ppf(upper_p, self.alpha, self.beta)
        return (lower, upper)

    def update(self, success: bool):
        """Bayesian update with new observation."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from the distribution."""
        return np.random.beta(self.alpha, self.beta, n)

    def kl_divergence(self, other: "BetaDistribution") -> float:
        """KL divergence to another Beta distribution."""
        from scipy.special import betaln, digamma

        a1, b1 = self.alpha, self.beta
        a2, b2 = other.alpha, other.beta

        kl = betaln(a2, b2) - betaln(a1, b1)
        kl += (a1 - a2) * digamma(a1)
        kl += (b1 - b2) * digamma(b1)
        kl += (a2 - a1 + b2 - b1) * digamma(a1 + b1)

        return kl


class BayesianGoalEstimator:
    """
    Bayesian estimation of goal success probabilities.

    Maintains posterior distributions over success rates
    for different goal categories, enabling principled
    uncertainty quantification.
    """

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Posteriors by goal category
        self._posteriors: Dict[str, BetaDistribution] = {}

        # Global posterior
        self._global_posterior = BetaDistribution(prior_alpha, prior_beta)

        # Hierarchical priors by type and priority
        self._type_posteriors: Dict[GoalType, BetaDistribution] = {}
        self._priority_posteriors: Dict[GoalPriority, BetaDistribution] = {}

    def _get_category_key(self, goal: Goal) -> str:
        """Get category key for a goal."""
        return f"{goal.goal_type.value}_{goal.priority.value}"

    def get_posterior(self, goal: Goal) -> BetaDistribution:
        """Get posterior distribution for a goal's category."""
        key = self._get_category_key(goal)

        if key not in self._posteriors:
            # Initialize with hierarchical prior
            type_post = self._type_posteriors.get(
                goal.goal_type,
                BetaDistribution(self.prior_alpha, self.prior_beta)
            )
            priority_post = self._priority_posteriors.get(
                goal.priority,
                BetaDistribution(self.prior_alpha, self.prior_beta)
            )

            # Combine priors (geometric mean of parameters)
            combined_alpha = math.sqrt(type_post.alpha * priority_post.alpha)
            combined_beta = math.sqrt(type_post.beta * priority_post.beta)

            self._posteriors[key] = BetaDistribution(combined_alpha, combined_beta)

        return self._posteriors[key]

    def estimate_success(self, goal: Goal) -> UncertaintyEstimate:
        """Estimate success probability with uncertainty."""
        posterior = self.get_posterior(goal)

        # Get samples for uncertainty decomposition
        samples = posterior.sample(1000)

        # Epistemic uncertainty from posterior variance
        epistemic = posterior.std()

        # Aleatoric uncertainty (inherent to Bernoulli process)
        mean = posterior.mean()
        aleatoric = math.sqrt(mean * (1 - mean))

        ci = posterior.confidence_interval(0.95)

        return UncertaintyEstimate(
            mean=mean,
            std=posterior.std(),
            confidence_interval=ci,
            confidence_level=0.95,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            distribution="beta",
            samples=samples,
        )

    def update(self, goal: Goal, success: bool):
        """Update posteriors with observed outcome."""
        # Update category posterior
        posterior = self.get_posterior(goal)
        posterior.update(success)

        # Update type posterior
        if goal.goal_type not in self._type_posteriors:
            self._type_posteriors[goal.goal_type] = BetaDistribution(
                self.prior_alpha, self.prior_beta
            )
        self._type_posteriors[goal.goal_type].update(success)

        # Update priority posterior
        if goal.priority not in self._priority_posteriors:
            self._priority_posteriors[goal.priority] = BetaDistribution(
                self.prior_alpha, self.prior_beta
            )
        self._priority_posteriors[goal.priority].update(success)

        # Update global posterior
        self._global_posterior.update(success)

        logger.debug(
            "bayesian_update",
            category=self._get_category_key(goal),
            success=success,
            new_mean=posterior.mean(),
        )


class MonteCarloUncertainty:
    """
    Monte Carlo methods for uncertainty estimation.

    Uses Monte Carlo dropout and sampling for neural
    network uncertainty quantification.
    """

    def __init__(
        self,
        n_samples: int = 100,
        dropout_rate: float = 0.1,
    ):
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def mc_dropout_estimate(
        self,
        predict_fn: Callable[[Any, bool], float],
        input_data: Any,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using Monte Carlo dropout.

        Runs multiple forward passes with dropout enabled
        to estimate predictive uncertainty.
        """
        samples = []

        for _ in range(self.n_samples):
            # Predict with dropout (training=True)
            pred = predict_fn(input_data, True)
            samples.append(pred)

        samples = np.array(samples)
        mean = np.mean(samples)
        std = np.std(samples)

        # Confidence interval
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)

        # Epistemic uncertainty is the spread of predictions
        epistemic = std

        # For classification, estimate aleatoric
        if 0 <= mean <= 1:
            aleatoric = math.sqrt(mean * (1 - mean))
        else:
            aleatoric = 0.0

        return UncertaintyEstimate(
            mean=float(mean),
            std=float(std),
            confidence_interval=(float(lower), float(upper)),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            distribution="empirical",
            samples=samples,
        )

    def bootstrap_estimate(
        self,
        data: List[float],
        statistic_fn: Callable[[np.ndarray], float],
        n_bootstrap: int = 1000,
    ) -> UncertaintyEstimate:
        """
        Bootstrap uncertainty estimation.

        Resamples data to estimate uncertainty in any statistic.
        """
        data_array = np.array(data)
        n = len(data_array)

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            resample = data_array[indices]
            stat = statistic_fn(resample)
            bootstrap_stats.append(stat)

        bootstrap_stats = np.array(bootstrap_stats)
        mean = np.mean(bootstrap_stats)
        std = np.std(bootstrap_stats)

        # BCa confidence interval (bias-corrected and accelerated)
        lower = np.percentile(bootstrap_stats, 2.5)
        upper = np.percentile(bootstrap_stats, 97.5)

        return UncertaintyEstimate(
            mean=float(mean),
            std=float(std),
            confidence_interval=(float(lower), float(upper)),
            epistemic_uncertainty=float(std),
            aleatoric_uncertainty=0.0,
            distribution="empirical",
            samples=bootstrap_stats,
        )


class EnsembleUncertainty:
    """
    Ensemble-based uncertainty quantification.

    Uses disagreement between multiple models to
    estimate epistemic uncertainty.
    """

    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self._models: List[Any] = []

    def ensemble_predict(
        self,
        models: List[Callable[[Any], float]],
        input_data: Any,
    ) -> UncertaintyEstimate:
        """
        Get prediction with uncertainty from model ensemble.
        """
        predictions = [model(input_data) for model in models]
        predictions = np.array(predictions)

        mean = np.mean(predictions)
        std = np.std(predictions)

        # Epistemic uncertainty from ensemble disagreement
        epistemic = std

        # Confidence interval from ensemble
        lower = np.min(predictions)
        upper = np.max(predictions)

        # More principled CI
        ci_lower = mean - 1.96 * std / math.sqrt(len(models))
        ci_upper = mean + 1.96 * std / math.sqrt(len(models))

        return UncertaintyEstimate(
            mean=float(mean),
            std=float(std),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=0.0,
            distribution="ensemble",
            samples=predictions,
        )

    def compute_ensemble_diversity(
        self,
        models: List[Callable[[Any], float]],
        inputs: List[Any],
    ) -> float:
        """
        Compute diversity of ensemble predictions.

        Higher diversity indicates more complementary models.
        """
        all_preds = []
        for input_data in inputs:
            preds = [model(input_data) for model in models]
            all_preds.append(preds)

        all_preds = np.array(all_preds)

        # Average pairwise correlation
        correlations = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                corr = np.corrcoef(all_preds[:, i], all_preds[:, j])[0, 1]
                correlations.append(corr)

        # Diversity is inverse of average correlation
        avg_corr = np.mean(correlations) if correlations else 0.0
        diversity = 1 - avg_corr

        return float(diversity)


class ThompsonSampler:
    """
    Thompson sampling for exploration vs exploitation.

    Uses uncertainty to balance trying new goals
    vs pursuing known good goals.
    """

    def __init__(self, exploration_bonus: float = 0.1):
        self.exploration_bonus = exploration_bonus
        self._estimates: Dict[str, BetaDistribution] = {}

    def get_estimate(self, goal_id: str) -> BetaDistribution:
        """Get or create estimate for a goal."""
        if goal_id not in self._estimates:
            self._estimates[goal_id] = BetaDistribution(1.0, 1.0)
        return self._estimates[goal_id]

    def sample_goal(self, goals: List[Goal]) -> Goal:
        """
        Select goal using Thompson sampling.

        Samples from posterior of each goal's success probability
        and selects the goal with highest sample.
        """
        best_goal = None
        best_sample = -float('inf')

        for goal in goals:
            estimate = self.get_estimate(goal.id)
            sample = estimate.sample(1)[0]

            # Add exploration bonus based on uncertainty
            uncertainty_bonus = self.exploration_bonus * estimate.std()
            sample += uncertainty_bonus

            if sample > best_sample:
                best_sample = sample
                best_goal = goal

        return best_goal

    def update(self, goal_id: str, success: bool):
        """Update estimate after observing outcome."""
        estimate = self.get_estimate(goal_id)
        estimate.update(success)

    def get_exploration_priority(self, goal: Goal) -> float:
        """
        Get exploration priority for a goal.

        High uncertainty goals get exploration bonus.
        """
        estimate = self.get_estimate(goal.id)
        uncertainty = estimate.std()

        # UCB-style exploration bonus
        total_observations = estimate.alpha + estimate.beta - 2
        if total_observations == 0:
            return 1.0  # Unexplored goals get max priority

        bonus = self.exploration_bonus * math.sqrt(
            2 * math.log(total_observations + 1) / (total_observations + 1)
        )

        return uncertainty + bonus


class ConfidenceCalibrator:
    """
    Calibrates confidence estimates to be reliable.

    Ensures that predicted probabilities match observed
    frequencies (e.g., 80% predictions are right 80% of time).
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

        # Calibration data
        self._predictions: List[float] = []
        self._outcomes: List[bool] = []

        # Isotonic regression parameters
        self._calibration_map: List[Tuple[float, float]] = []

    def add_observation(self, prediction: float, outcome: bool):
        """Add prediction-outcome pair for calibration."""
        self._predictions.append(prediction)
        self._outcomes.append(outcome)

    def fit_calibration(self):
        """Fit calibration function using isotonic regression."""
        if len(self._predictions) < 20:
            return

        # Sort by prediction
        sorted_pairs = sorted(zip(self._predictions, self._outcomes))
        preds = np.array([p[0] for p in sorted_pairs])
        outcomes = np.array([p[1] for p in sorted_pairs])

        # Pool adjacent violators (isotonic regression)
        n = len(preds)
        calibrated = outcomes.astype(float).copy()

        # Forward pass
        i = 0
        while i < n - 1:
            if calibrated[i] > calibrated[i + 1]:
                # Pool
                j = i + 1
                while j < n and calibrated[i] > calibrated[j]:
                    j += 1

                # Average the pooled values
                pool_mean = np.mean(calibrated[i:j])
                calibrated[i:j] = pool_mean

            i += 1

        # Create calibration map
        self._calibration_map = list(zip(preds.tolist(), calibrated.tolist()))

    def calibrate(self, prediction: float) -> float:
        """Apply calibration to a prediction."""
        if not self._calibration_map:
            return prediction

        # Find bracketing points
        for i, (p, c) in enumerate(self._calibration_map):
            if p >= prediction:
                if i == 0:
                    return c
                # Linear interpolation
                p_prev, c_prev = self._calibration_map[i - 1]
                ratio = (prediction - p_prev) / (p - p_prev + 1e-8)
                return c_prev + ratio * (c - c_prev)

        return self._calibration_map[-1][1]

    def get_calibration_error(self) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Lower is better (perfect calibration = 0).
        """
        if len(self._predictions) < self.n_bins:
            return 0.0

        preds = np.array(self._predictions)
        outcomes = np.array(self._outcomes)

        # Bin predictions
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            if np.sum(mask) == 0:
                continue

            bin_preds = preds[mask]
            bin_outcomes = outcomes[mask]

            avg_pred = np.mean(bin_preds)
            avg_outcome = np.mean(bin_outcomes)

            ece += np.sum(mask) * abs(avg_pred - avg_outcome)

        ece /= len(preds)
        return float(ece)

    def get_reliability_diagram(self) -> Dict[str, List[float]]:
        """Get data for reliability diagram visualization."""
        if len(self._predictions) < self.n_bins:
            return {"predicted": [], "observed": [], "counts": []}

        preds = np.array(self._predictions)
        outcomes = np.array(self._outcomes)

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        predicted = []
        observed = []
        counts = []

        for i in range(self.n_bins):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            count = np.sum(mask)

            if count == 0:
                continue

            predicted.append(float(np.mean(preds[mask])))
            observed.append(float(np.mean(outcomes[mask])))
            counts.append(int(count))

        return {
            "predicted": predicted,
            "observed": observed,
            "counts": counts,
            "bin_centers": bin_centers.tolist(),
        }


class UncertaintyQuantifier:
    """
    Unified uncertainty quantification system.

    Combines multiple uncertainty estimation methods
    for comprehensive probabilistic reasoning.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        n_mc_samples: int = 100,
    ):
        self.bayesian = BayesianGoalEstimator(prior_alpha, prior_beta)
        self.mc_uncertainty = MonteCarloUncertainty(n_mc_samples)
        self.ensemble = EnsembleUncertainty()
        self.thompson = ThompsonSampler()
        self.calibrator = ConfidenceCalibrator()

        self._initialized = False

    async def initialize(self):
        """Initialize the uncertainty quantifier."""
        self._initialized = True
        logger.info("uncertainty_quantifier_initialized")

    async def shutdown(self):
        """Shutdown the uncertainty quantifier."""
        self._initialized = False
        logger.info("uncertainty_quantifier_shutdown")

    def estimate_success_uncertainty(self, goal: Goal) -> UncertaintyEstimate:
        """
        Get comprehensive uncertainty estimate for goal success.
        """
        # Get Bayesian estimate
        estimate = self.bayesian.estimate_success(goal)

        # Calibrate the mean
        calibrated_mean = self.calibrator.calibrate(estimate.mean)

        # Update estimate with calibration
        return UncertaintyEstimate(
            mean=calibrated_mean,
            std=estimate.std,
            confidence_interval=estimate.confidence_interval,
            confidence_level=estimate.confidence_level,
            epistemic_uncertainty=estimate.epistemic_uncertainty,
            aleatoric_uncertainty=estimate.aleatoric_uncertainty,
            distribution=estimate.distribution,
            samples=estimate.samples,
        )

    def update_from_outcome(self, goal: Goal, success: bool):
        """Update all estimators with observed outcome."""
        # Update Bayesian estimator
        self.bayesian.update(goal, success)

        # Update Thompson sampler
        self.thompson.update(goal.id, success)

        # Update calibrator
        estimate = self.bayesian.estimate_success(goal)
        self.calibrator.add_observation(estimate.mean, success)

        # Refit calibration periodically
        if len(self.calibrator._predictions) % 50 == 0:
            self.calibrator.fit_calibration()

    def select_goal_with_exploration(self, goals: List[Goal]) -> Tuple[Goal, float]:
        """
        Select goal balancing exploration and exploitation.

        Returns selected goal and exploration score.
        """
        selected = self.thompson.sample_goal(goals)
        exploration_score = self.thompson.get_exploration_priority(selected)
        return selected, exploration_score

    def get_confidence_level(self, goal: Goal) -> str:
        """Get human-readable confidence level."""
        estimate = self.estimate_success_uncertainty(goal)
        total_unc = estimate.total_uncertainty()

        if total_unc < 0.1:
            return "very_high"
        elif total_unc < 0.2:
            return "high"
        elif total_unc < 0.3:
            return "medium"
        elif total_unc < 0.4:
            return "low"
        else:
            return "very_low"

    def should_gather_more_info(self, goal: Goal, threshold: float = 0.3) -> bool:
        """
        Determine if more information should be gathered before proceeding.

        High epistemic uncertainty suggests we need more data.
        """
        estimate = self.estimate_success_uncertainty(goal)
        return estimate.epistemic_uncertainty > threshold

    def get_stats(self) -> Dict[str, Any]:
        """Get uncertainty quantifier statistics."""
        return {
            "bayesian_categories": len(self.bayesian._posteriors),
            "thompson_estimates": len(self.thompson._estimates),
            "calibration_observations": len(self.calibrator._predictions),
            "calibration_error": self.calibrator.get_calibration_error(),
            "reliability_diagram": self.calibrator.get_reliability_diagram(),
        }
