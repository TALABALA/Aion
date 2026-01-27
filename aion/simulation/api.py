"""AION Simulation API - Programmatic interface for running simulations.

Provides a high-level, ergonomic API for:
- Creating and running simulations.
- Generating scenarios from templates.
- Evaluating results.
- Running adversarial test suites.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.simulation.config import SimulationEnvironmentConfig, default_config, test_config
from aion.simulation.environment import SimulationEnvironment
from aion.simulation.evaluation.evaluator import EvaluationResult
from aion.simulation.scenarios.templates import ScenarioTemplateLibrary
from aion.simulation.types import (
    Assertion,
    EvaluationMetric,
    Scenario,
    ScenarioType,
    SimulationConfig,
    SimulationResult,
    TimeMode,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class SimulationAPI:
    """High-level API for the simulation environment.

    Usage:
        api = SimulationAPI()

        # Quick run
        result = await api.run_scenario("customer_support_basic")

        # With custom config
        result = await api.run_scenario(
            "task_execution_pipeline",
            sim_config=SimulationConfig(max_ticks=500, seed=42),
        )

        # Adversarial testing
        results = await api.run_adversarial_suite(coverage="comprehensive")

        # Evaluate
        eval_result = await api.evaluate_result(result)
    """

    def __init__(
        self,
        kernel: Optional["AIONKernel"] = None,
        env_config: Optional[SimulationEnvironmentConfig] = None,
    ) -> None:
        self.kernel = kernel
        self.env_config = env_config or default_config()
        self.templates = ScenarioTemplateLibrary()
        self._env: Optional[SimulationEnvironment] = None

    def _get_env(self) -> SimulationEnvironment:
        if self._env is None:
            self._env = SimulationEnvironment(
                kernel=self.kernel,
                env_config=self.env_config,
            )
        return self._env

    # -- Quick Run --

    async def run_scenario(
        self,
        template_name: str,
        overrides: Optional[Dict[str, Any]] = None,
        sim_config: Optional[SimulationConfig] = None,
        agent_id: Optional[str] = None,
        tool_mocks: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """Run a scenario from a template.

        Args:
            template_name: Name of the scenario template.
            overrides: Template parameter overrides.
            sim_config: Simulation configuration.
            agent_id: Optional agent to load.
            tool_mocks: Optional tool mocks for agent.

        Returns:
            Simulation result.
        """
        scenario = self.templates.instantiate(template_name, overrides)
        env = self._get_env()

        await env.create_simulation(scenario, sim_config)

        if agent_id:
            await env.load_agent(agent_id, tool_mocks=tool_mocks)

        return await env.run()

    async def run_custom_scenario(
        self,
        scenario: Scenario,
        sim_config: Optional[SimulationConfig] = None,
        agent_id: Optional[str] = None,
        tool_mocks: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """Run a custom scenario."""
        env = self._get_env()
        await env.create_simulation(scenario, sim_config)

        if agent_id:
            await env.load_agent(agent_id, tool_mocks=tool_mocks)

        return await env.run()

    # -- Batch --

    async def run_template_batch(
        self,
        template_names: List[str],
        sim_config: Optional[SimulationConfig] = None,
    ) -> List[SimulationResult]:
        """Run multiple templates."""
        scenarios = [self.templates.instantiate(name) for name in template_names]
        env = self._get_env()
        return await env.run_scenarios(scenarios, sim_config)

    async def run_generated_scenarios(
        self,
        scenario_type: ScenarioType,
        count: int = 5,
        sim_config: Optional[SimulationConfig] = None,
    ) -> List[SimulationResult]:
        """Generate and run scenarios."""
        env = self._get_env()
        scenarios = []
        for _ in range(count):
            scenario = await env.scenario_generator.generate(scenario_type)
            scenarios.append(scenario)
        return await env.run_scenarios(scenarios, sim_config)

    # -- Agent Testing --

    async def test_agent(
        self,
        agent_id: str,
        agent_config: Dict[str, Any],
        tool_mocks: Optional[Dict[str, Any]] = None,
        coverage: str = "standard",
        sim_config: Optional[SimulationConfig] = None,
    ) -> Dict[str, Any]:
        """Run a comprehensive test suite for an agent.

        Returns:
            Test results including pass/fail, metrics, and evaluation.
        """
        env = self._get_env()

        # Generate scenarios for agent
        scenarios = await env.scenario_generator.generate_for_agent(
            agent_config, test_coverage=coverage,
        )

        results: List[SimulationResult] = []
        evaluations: List[EvaluationResult] = []

        for scenario in scenarios:
            await env.create_simulation(scenario, sim_config)
            await env.load_agent(agent_id, config=agent_config, tool_mocks=tool_mocks)
            result = await env.run()
            results.append(result)

            evaluation = await env.evaluate()
            evaluations.append(evaluation)

        passed = sum(1 for e in evaluations if e.passed)
        total = len(evaluations)

        return {
            "agent_id": agent_id,
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "results": results,
            "evaluations": evaluations,
            "coverage": coverage,
        }

    # -- Adversarial --

    async def run_adversarial_suite(
        self,
        coverage: str = "standard",
        sim_config: Optional[SimulationConfig] = None,
    ) -> List[SimulationResult]:
        """Run adversarial test suite."""
        env = self._get_env()
        scenarios = env.adversarial.generate_suite(coverage=coverage)
        return await env.run_scenarios(scenarios, sim_config)

    # -- Evaluation --

    async def evaluate_result(
        self,
        result: SimulationResult,
    ) -> EvaluationResult:
        """Evaluate a simulation result."""
        env = self._get_env()
        final_state = result.final_state
        if final_state is None:
            from aion.simulation.types import WorldState
            final_state = WorldState()
        return await env.evaluator.evaluate(result, final_state)

    async def compare_results(
        self,
        results: List[SimulationResult],
    ) -> Dict[str, Any]:
        """Compare multiple simulation results."""
        env = self._get_env()
        return await env.evaluator.compare_runs(results)

    # -- Configuration --

    def add_assertion(self, assertion: Assertion) -> None:
        env = self._get_env()
        env.evaluator.add_assertion(assertion)

    def add_metric(self, metric: EvaluationMetric) -> None:
        env = self._get_env()
        env.evaluator.register_metric(metric)

    def register_template(self, name: str, template: Dict[str, Any]) -> None:
        self.templates.register(name, template)

    def list_templates(self) -> List[str]:
        return self.templates.list_templates()
