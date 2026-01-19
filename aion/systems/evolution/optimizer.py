"""
AION Parameter Optimizer

Bayesian and gradient-based optimization for system parameters:
- Multi-objective optimization
- Constraint handling
- Safe exploration
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OptimizationBounds:
    """Bounds for an optimization parameter."""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None  # Discrete step size
    log_scale: bool = False  # Use log scale for optimization


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_params: dict[str, float]
    best_score: float
    iterations: int
    history: list[tuple[dict[str, float], float]]
    converged: bool


class ParameterOptimizer:
    """
    Multi-strategy parameter optimizer.

    Supports:
    - Random search
    - Grid search
    - Bayesian optimization (simplified)
    - Gradient-free optimization
    """

    def __init__(
        self,
        objective_fn: Callable[[dict[str, float]], float],
        bounds: list[OptimizationBounds],
        maximize: bool = True,
        seed: Optional[int] = None,
    ):
        self.objective_fn = objective_fn
        self.bounds = {b.name: b for b in bounds}
        self.maximize = maximize

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # History
        self._history: list[tuple[dict[str, float], float]] = []
        self._best_params: Optional[dict[str, float]] = None
        self._best_score: Optional[float] = None

    def _sample_point(self) -> dict[str, float]:
        """Sample a random point in the parameter space."""
        point = {}
        for name, b in self.bounds.items():
            if b.log_scale:
                log_val = random.uniform(math.log(b.min_value), math.log(b.max_value))
                val = math.exp(log_val)
            else:
                val = random.uniform(b.min_value, b.max_value)

            if b.step is not None:
                val = b.min_value + round((val - b.min_value) / b.step) * b.step

            point[name] = val

        return point

    def _evaluate(self, params: dict[str, float]) -> float:
        """Evaluate the objective function."""
        try:
            score = self.objective_fn(params)
            self._history.append((params.copy(), score))

            # Update best
            if self._best_score is None:
                self._best_score = score
                self._best_params = params.copy()
            elif self.maximize and score > self._best_score:
                self._best_score = score
                self._best_params = params.copy()
            elif not self.maximize and score < self._best_score:
                self._best_score = score
                self._best_params = params.copy()

            return score

        except Exception as e:
            logger.warning(f"Objective evaluation failed: {e}")
            return float("-inf") if self.maximize else float("inf")

    def random_search(
        self,
        n_iterations: int = 100,
    ) -> OptimizationResult:
        """
        Perform random search optimization.

        Args:
            n_iterations: Number of iterations

        Returns:
            OptimizationResult
        """
        logger.info("Starting random search", iterations=n_iterations)

        for i in range(n_iterations):
            params = self._sample_point()
            self._evaluate(params)

        return OptimizationResult(
            best_params=self._best_params or {},
            best_score=self._best_score or 0.0,
            iterations=n_iterations,
            history=self._history.copy(),
            converged=True,
        )

    def grid_search(
        self,
        n_points_per_dim: int = 10,
    ) -> OptimizationResult:
        """
        Perform grid search optimization.

        Args:
            n_points_per_dim: Number of points per dimension

        Returns:
            OptimizationResult
        """
        logger.info("Starting grid search", points_per_dim=n_points_per_dim)

        # Generate grid
        grids = []
        for name, b in self.bounds.items():
            if b.log_scale:
                points = np.logspace(
                    math.log10(b.min_value),
                    math.log10(b.max_value),
                    n_points_per_dim,
                )
            else:
                points = np.linspace(b.min_value, b.max_value, n_points_per_dim)
            grids.append((name, points))

        # Evaluate grid points
        def eval_grid(idx, current_params):
            if idx == len(grids):
                self._evaluate(current_params)
                return

            name, points = grids[idx]
            for val in points:
                current_params[name] = val
                eval_grid(idx + 1, current_params)

        eval_grid(0, {})

        return OptimizationResult(
            best_params=self._best_params or {},
            best_score=self._best_score or 0.0,
            iterations=len(self._history),
            history=self._history.copy(),
            converged=True,
        )

    def bayesian_optimize(
        self,
        n_iterations: int = 50,
        n_initial: int = 10,
    ) -> OptimizationResult:
        """
        Perform simplified Bayesian optimization.

        Uses a surrogate model to guide the search.

        Args:
            n_iterations: Total iterations
            n_initial: Initial random samples

        Returns:
            OptimizationResult
        """
        logger.info("Starting Bayesian optimization", iterations=n_iterations)

        # Initial random samples
        for _ in range(n_initial):
            params = self._sample_point()
            self._evaluate(params)

        # Optimize using surrogate model
        for i in range(n_iterations - n_initial):
            # Build simple surrogate (weighted nearest neighbors)
            params = self._acquisition_sample()
            self._evaluate(params)

        return OptimizationResult(
            best_params=self._best_params or {},
            best_score=self._best_score or 0.0,
            iterations=n_iterations,
            history=self._history.copy(),
            converged=True,
        )

    def _acquisition_sample(self) -> dict[str, float]:
        """Sample using acquisition function (simplified UCB)."""
        if len(self._history) < 5:
            return self._sample_point()

        # Sample candidates
        candidates = [self._sample_point() for _ in range(100)]

        # Score candidates using UCB
        scores = []
        for c in candidates:
            # Estimate mean and variance from nearest neighbors
            distances = []
            values = []

            for h_params, h_score in self._history:
                dist = sum(
                    ((c[k] - h_params[k]) / (self.bounds[k].max_value - self.bounds[k].min_value)) ** 2
                    for k in c
                )
                distances.append(dist)
                values.append(h_score)

            # Weighted by inverse distance
            weights = [1.0 / (d + 0.01) for d in distances]
            total_weight = sum(weights)
            mean = sum(w * v for w, v in zip(weights, values)) / total_weight
            variance = sum(w * (v - mean) ** 2 for w, v in zip(weights, values)) / total_weight

            # UCB score
            exploration_weight = 1.0
            ucb = mean + exploration_weight * math.sqrt(variance)
            scores.append(ucb if self.maximize else -ucb)

        # Return best candidate
        best_idx = scores.index(max(scores))
        return candidates[best_idx]

    def nelder_mead(
        self,
        n_iterations: int = 100,
        initial_params: Optional[dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Perform Nelder-Mead simplex optimization.

        Args:
            n_iterations: Maximum iterations
            initial_params: Starting point

        Returns:
            OptimizationResult
        """
        logger.info("Starting Nelder-Mead optimization", iterations=n_iterations)

        n_dims = len(self.bounds)

        # Initialize simplex
        if initial_params:
            x0 = np.array([initial_params[k] for k in sorted(self.bounds.keys())])
        else:
            x0 = np.array([
                (b.min_value + b.max_value) / 2
                for _, b in sorted(self.bounds.items())
            ])

        # Create initial simplex
        simplex = [x0]
        for i in range(n_dims):
            point = x0.copy()
            b = list(self.bounds.values())[i]
            point[i] += (b.max_value - b.min_value) * 0.1
            simplex.append(point)

        def to_dict(arr):
            return dict(zip(sorted(self.bounds.keys()), arr))

        def clip_to_bounds(arr):
            result = arr.copy()
            for i, (name, b) in enumerate(sorted(self.bounds.items())):
                result[i] = max(b.min_value, min(b.max_value, result[i]))
            return result

        # Evaluate simplex
        values = [self._evaluate(to_dict(clip_to_bounds(p))) for p in simplex]

        # Optimization loop
        for iteration in range(n_iterations):
            # Sort simplex
            order = np.argsort(values if not self.maximize else [-v for v in values])
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]

            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            reflected = centroid + (centroid - simplex[-1])
            reflected = clip_to_bounds(reflected)
            fr = self._evaluate(to_dict(reflected))

            if (self.maximize and values[0] <= fr < values[-2]) or \
               (not self.maximize and values[0] >= fr > values[-2]):
                simplex[-1] = reflected
                values[-1] = fr
                continue

            # Expansion
            if (self.maximize and fr >= values[0]) or (not self.maximize and fr <= values[0]):
                expanded = centroid + 2 * (reflected - centroid)
                expanded = clip_to_bounds(expanded)
                fe = self._evaluate(to_dict(expanded))

                if (self.maximize and fe > fr) or (not self.maximize and fe < fr):
                    simplex[-1] = expanded
                    values[-1] = fe
                else:
                    simplex[-1] = reflected
                    values[-1] = fr
                continue

            # Contraction
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            contracted = clip_to_bounds(contracted)
            fc = self._evaluate(to_dict(contracted))

            if (self.maximize and fc > values[-1]) or (not self.maximize and fc < values[-1]):
                simplex[-1] = contracted
                values[-1] = fc
                continue

            # Shrink
            for i in range(1, len(simplex)):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                simplex[i] = clip_to_bounds(simplex[i])
                values[i] = self._evaluate(to_dict(simplex[i]))

        return OptimizationResult(
            best_params=self._best_params or {},
            best_score=self._best_score or 0.0,
            iterations=len(self._history),
            history=self._history.copy(),
            converged=True,
        )

    def get_best(self) -> tuple[Optional[dict[str, float]], Optional[float]]:
        """Get the best parameters and score found."""
        return self._best_params, self._best_score

    def reset(self) -> None:
        """Reset the optimizer state."""
        self._history.clear()
        self._best_params = None
        self._best_score = None
