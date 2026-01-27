"""AION Simulation Evaluator - Evaluate simulation results with statistical rigor.

Provides:
- EvaluationResult: Structured evaluation outcome.
- SimulationEvaluator: Evaluates simulations using metrics, assertions,
  scoring functions, and multi-run statistical comparison.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.simulation.evaluation.metrics import MetricsCollector
from aion.simulation.types import (
    Assertion,
    EvaluationMetric,
    SimulationResult,
    SimulationStatus,
    WorldState,
)

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of simulation evaluation."""

    simulation_id: str = ""

    # Overall
    passed: bool = False
    score: float = 0.0

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Assertions
    assertions_passed: int = 0
    assertions_failed: int = 0
    assertion_results: List[Dict[str, Any]] = field(default_factory=list)

    # Comparison (if comparing runs)
    comparison: Optional[Dict[str, Any]] = None

    # Report
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Statistical
    confidence: float = 0.0
    metric_summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)


class SimulationEvaluator:
    """Evaluates simulation runs with statistical rigor.

    SOTA features:
    - Custom metric calculators with aggregation.
    - Assertion checking with severity levels and context.
    - Weighted scoring with configurable thresholds.
    - Multi-run comparison with Welch's t-test.
    - Percentile-based performance evaluation.
    - Comprehensive report generation.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, EvaluationMetric] = {}
        self._assertions: List[Assertion] = []
        self._scoring_fn: Optional[Callable] = None
        self.metrics_collector = MetricsCollector()
        self._register_builtin_metrics()

    def _register_builtin_metrics(self) -> None:
        self.register_metric(EvaluationMetric(
            name="total_actions",
            description="Total actions taken",
            aggregation="sum",
        ))
        self.register_metric(EvaluationMetric(
            name="success_rate",
            description="Percentage of successful actions",
            aggregation="mean",
            min_threshold=0.0,
        ))
        self.register_metric(EvaluationMetric(
            name="avg_response_time",
            description="Average response time in ms",
            aggregation="mean",
        ))
        self.register_metric(EvaluationMetric(
            name="error_count",
            description="Number of errors",
            aggregation="sum",
        ))
        self.register_metric(EvaluationMetric(
            name="goal_completion_rate",
            description="Ratio of goals achieved",
            aggregation="mean",
            min_threshold=0.5,
            weight=2.0,
        ))

    def register_metric(self, metric: EvaluationMetric) -> None:
        self._metrics[metric.name] = metric

    def add_assertion(self, assertion: Assertion) -> None:
        self._assertions.append(assertion)

    def set_scoring_function(self, fn: Callable) -> None:
        """Set a custom scoring function: (metrics, result) -> float."""
        self._scoring_fn = fn

    # -- Evaluation --

    async def evaluate(
        self,
        result: SimulationResult,
        final_state: WorldState,
    ) -> EvaluationResult:
        """Evaluate a simulation result."""
        eval_result = EvaluationResult(simulation_id=result.simulation_id)

        # Calculate metrics
        eval_result.metrics = await self._calculate_metrics(result, final_state)

        # Record metrics for statistical analysis
        for name, value in eval_result.metrics.items():
            self.metrics_collector.record(name, value, tick=result.total_ticks)

        # Metric summaries
        eval_result.metric_summaries = {
            name: {
                "value": value,
                "target": self._metrics[name].target if name in self._metrics else None,
                "threshold_met": self._check_threshold(name, value),
            }
            for name, value in eval_result.metrics.items()
        }

        # Check assertions
        assertion_results = await self._check_assertions(result, final_state)
        eval_result.assertion_results = assertion_results
        eval_result.assertions_passed = sum(1 for a in assertion_results if a["passed"])
        eval_result.assertions_failed = sum(1 for a in assertion_results if not a["passed"])

        # Calculate score
        if self._scoring_fn:
            eval_result.score = self._scoring_fn(eval_result.metrics, result)
        else:
            eval_result.score = self._calculate_score(eval_result)

        # Determine pass/fail
        has_critical_failures = any(
            not a["passed"] and a["severity"] == "error"
            for a in assertion_results
        )
        eval_result.passed = (
            not has_critical_failures
            and result.status == SimulationStatus.COMPLETED
            and eval_result.score >= 0.5
        )

        # Generate summary
        eval_result.summary = self._generate_summary(eval_result, result)

        # Details
        eval_result.details = {
            "status": result.status.value,
            "goals_achieved": result.goals_achieved,
            "goals_failed": result.goals_failed,
            "errors": result.errors,
            "total_ticks": result.total_ticks,
            "real_time_seconds": result.total_real_time,
        }

        return eval_result

    async def _calculate_metrics(
        self,
        result: SimulationResult,
        final_state: WorldState,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for name, metric in self._metrics.items():
            if metric.calculator:
                try:
                    value = metric.calculator(result, final_state)
                except Exception as exc:
                    logger.warning("metric_calc_error", metric=name, error=str(exc))
                    continue
            else:
                value = self._calculate_builtin(name, result, final_state)

            if value is not None:
                metrics[name] = value

        return metrics

    def _calculate_builtin(
        self,
        name: str,
        result: SimulationResult,
        state: WorldState,
    ) -> Optional[float]:
        if name == "total_actions":
            return float(result.event_count)
        elif name == "success_rate":
            achieved = len(result.goals_achieved)
            failed = len(result.goals_failed)
            total = achieved + failed
            return achieved / total if total > 0 else 1.0
        elif name == "avg_response_time":
            if result.total_ticks == 0:
                return 0.0
            return result.total_real_time / result.total_ticks * 1000
        elif name == "error_count":
            return float(len(result.errors))
        elif name == "goal_completion_rate":
            achieved = len(result.goals_achieved)
            failed = len(result.goals_failed)
            total = achieved + failed
            return achieved / total if total > 0 else (1.0 if not result.goals_failed else 0.0)
        return None

    def _check_threshold(self, name: str, value: float) -> bool:
        metric = self._metrics.get(name)
        if metric is None:
            return True
        if metric.min_threshold is not None and value < metric.min_threshold:
            return False
        if metric.max_threshold is not None and value > metric.max_threshold:
            return False
        return True

    async def _check_assertions(
        self,
        result: SimulationResult,
        final_state: WorldState,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for assertion in self._assertions:
            passed = self._evaluate_assertion(assertion, result, final_state)
            entry = {
                "id": assertion.id,
                "name": assertion.name,
                "passed": passed,
                "severity": assertion.severity,
                "message": "" if passed else (assertion.message or f"Assertion '{assertion.name}' failed"),
            }
            results.append(entry)
            assertion.passed = passed

        return results

    def _evaluate_assertion(
        self,
        assertion: Assertion,
        result: SimulationResult,
        state: WorldState,
    ) -> bool:
        # Use compiled function if available
        if assertion.condition_fn:
            try:
                return assertion.condition_fn(result, state)
            except Exception:
                return False

        context = {
            "result": result,
            "state": state,
            "goals_achieved": len(result.goals_achieved),
            "goals_failed": len(result.goals_failed),
            "errors": len(result.errors),
            "ticks": result.total_ticks,
            "status": result.status.value,
            "success": result.success,
            "event_count": result.event_count,
        }

        try:
            condition = assertion.condition

            for op in (">=", "<=", "!=", "==", ">", "<"):
                if op in condition:
                    parts = condition.split(op, 1)
                    left = self._resolve_value(parts[0].strip(), context)
                    right = self._resolve_value(parts[1].strip(), context)
                    if op == "==":
                        return left == right
                    elif op == "!=":
                        return left != right
                    elif op == ">=":
                        return float(left) >= float(right)
                    elif op == "<=":
                        return float(left) <= float(right)
                    elif op == ">":
                        return float(left) > float(right)
                    elif op == "<":
                        return float(left) < float(right)

            # Boolean context values
            if condition in context:
                return bool(context[condition])

            return True

        except Exception as exc:
            logger.warning("assertion_eval_error", name=assertion.name, error=str(exc))
            return False

    def _resolve_value(self, expr: str, context: Dict) -> Any:
        expr = expr.strip("'\"")
        if expr in context:
            return context[expr]
        try:
            return float(expr)
        except ValueError:
            return expr

    def _calculate_score(self, eval_result: EvaluationResult) -> float:
        scores: List[float] = []
        weights: List[float] = []

        # Assertion score
        total_assertions = eval_result.assertions_passed + eval_result.assertions_failed
        if total_assertions > 0:
            assertion_score = eval_result.assertions_passed / total_assertions
            scores.append(assertion_score)
            weights.append(2.0)

        # Metric scores
        for name, metric in self._metrics.items():
            value = eval_result.metrics.get(name)
            if value is None:
                continue

            if metric.target is not None:
                diff = abs(value - metric.target)
                max_diff = abs(metric.target) + 1
                metric_score = max(0.0, 1.0 - diff / max_diff)
            elif metric.min_threshold is not None:
                metric_score = 1.0 if value >= metric.min_threshold else 0.0
            elif metric.max_threshold is not None:
                metric_score = 1.0 if value <= metric.max_threshold else 0.0
            else:
                continue

            scores.append(metric_score)
            weights.append(metric.weight)

        if not scores:
            return 0.5

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _generate_summary(
        self,
        eval_result: EvaluationResult,
        sim_result: SimulationResult,
    ) -> str:
        verdict = "PASSED" if eval_result.passed else "FAILED"
        lines = [
            f"Simulation Evaluation: {verdict}",
            f"Score: {eval_result.score:.3f}",
            f"Status: {sim_result.status.value}",
            f"Duration: {sim_result.total_ticks} ticks "
            f"({sim_result.total_simulation_time:.1f}s sim, "
            f"{sim_result.total_real_time:.2f}s real)",
            f"Assertions: {eval_result.assertions_passed}/"
            f"{eval_result.assertions_passed + eval_result.assertions_failed} passed",
        ]

        if eval_result.assertions_failed > 0:
            lines.append("\nFailed Assertions:")
            for a in eval_result.assertion_results:
                if not a["passed"]:
                    lines.append(f"  [{a['severity'].upper()}] {a['name']}: {a['message']}")

        lines.append("\nMetrics:")
        for name, value in sorted(eval_result.metrics.items()):
            threshold_ok = eval_result.metric_summaries.get(name, {}).get("threshold_met", True)
            marker = " OK" if threshold_ok else " FAIL"
            lines.append(f"  {name}: {value:.4f}{marker}")

        return "\n".join(lines)

    # -- Multi-Run Comparison --

    async def compare_runs(
        self,
        results: List[SimulationResult],
    ) -> Dict[str, Any]:
        if len(results) < 2:
            return {"error": "Need at least 2 results to compare"}

        comparison: Dict[str, Any] = {
            "run_count": len(results),
            "metrics": {},
            "success_rate": sum(1 for r in results if r.success) / len(results),
        }

        all_metrics: Dict[str, List[float]] = {}
        for result in results:
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        for key, values in all_metrics.items():
            n = len(values)
            mean = statistics.mean(values) if values else 0.0
            std = statistics.stdev(values) if n > 1 else 0.0
            sorted_v = sorted(values)
            comparison["metrics"][key] = {
                "min": min(values),
                "max": max(values),
                "mean": mean,
                "median": statistics.median(values),
                "stddev": std,
                "cv": std / mean if mean != 0 else 0.0,
                "p95": sorted_v[int(0.95 * (n - 1))] if n > 1 else mean,
            }

        return comparison

    async def a_b_compare(
        self,
        results_a: List[SimulationResult],
        results_b: List[SimulationResult],
    ) -> Dict[str, Any]:
        """Compare two groups of runs (A/B testing)."""
        collector_a = MetricsCollector()
        collector_b = MetricsCollector()

        for r in results_a:
            for k, v in r.metrics.items():
                collector_a.record(k, v)
        for r in results_b:
            for k, v in r.metrics.items():
                collector_b.record(k, v)

        all_keys = set(collector_a.metric_names()) | set(collector_b.metric_names())
        comparisons: Dict[str, Any] = {}
        for key in all_keys:
            comparisons[key] = collector_a.compare(key, collector_b)

        return {
            "group_a_runs": len(results_a),
            "group_b_runs": len(results_b),
            "metric_comparisons": comparisons,
            "any_significant": any(
                c.get("significant", False) for c in comparisons.values()
            ),
        }

    # -- Cleanup --

    def clear_assertions(self) -> None:
        self._assertions.clear()

    def reset(self) -> None:
        self._assertions.clear()
        self.metrics_collector.clear()
