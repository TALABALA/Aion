"""AION Adversarial Generator - Orchestrates adversarial test generation.

Provides:
- AdversarialGenerator: Top-level orchestrator combining fuzzing,
  edge case discovery, and stress testing into comprehensive
  adversarial test suites.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import structlog

from aion.simulation.adversarial.edge_cases import EdgeCaseDiscovery
from aion.simulation.adversarial.fuzzing import Fuzzer
from aion.simulation.adversarial.stress import StressTestGenerator
from aion.simulation.types import FuzzStrategy, Scenario, ScenarioType

logger = structlog.get_logger(__name__)


class AdversarialGenerator:
    """Orchestrates adversarial test generation.

    Combines:
    - Edge case discovery (systematic boundary/security exploration).
    - Mutation-based fuzzing (random input generation).
    - Stress testing (load profile generation).

    Generates comprehensive adversarial test suites for agents.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.edge_cases = EdgeCaseDiscovery()
        self.fuzzer = Fuzzer(strategy=FuzzStrategy.MUTATION, seed=seed)
        self.stress = StressTestGenerator()

        self._generated_scenarios: List[Scenario] = []

    # -- Full Suite --

    def generate_suite(
        self,
        agent_config: Optional[Dict[str, Any]] = None,
        coverage: str = "standard",
    ) -> List[Scenario]:
        """Generate a comprehensive adversarial test suite.

        Args:
            agent_config: Agent capabilities for targeted generation.
            coverage: Level ('minimal', 'standard', 'comprehensive').

        Returns:
            List of adversarial scenarios.
        """
        scenarios: List[Scenario] = []

        if coverage in ("minimal", "standard", "comprehensive"):
            # Edge cases
            edge_cases = self.edge_cases.discover_category("boundary")
            edge_cases.extend(self.edge_cases.discover_category("injection"))
            scenarios.extend(self.edge_cases.to_scenarios(edge_cases))

        if coverage in ("standard", "comprehensive"):
            # Fuzzing
            base_input = {"message": "Hello, please help me"}
            fuzzed = self.fuzzer.generate(base_input=base_input, count=20)
            for i, fuzz_input in enumerate(fuzzed):
                scenarios.append(Scenario(
                    name=f"fuzz_{i}",
                    description=f"Fuzzed input #{i}",
                    type=ScenarioType.ADVERSARIAL,
                    initial_entities=[{"type": "user", "name": "fuzz_user"}],
                    scripted_events=[
                        {"time": 0, "type": "user_input", "data": fuzz_input},
                    ],
                    success_criteria=[
                        {"name": "no_crash", "condition": "simulation_completed"},
                    ],
                    tags={"fuzz", "adversarial"},
                    difficulty=0.6,
                ))

            # More edge categories
            for cat in ["encoding", "type_coercion", "concurrency"]:
                cases = self.edge_cases.discover_category(cat)
                scenarios.extend(self.edge_cases.to_scenarios(cases))

        if coverage == "comprehensive":
            # All remaining edge case categories
            for cat in ["resource", "state"]:
                cases = self.edge_cases.discover_category(cat)
                scenarios.extend(self.edge_cases.to_scenarios(cases))

            # Evolutionary fuzzing
            evo_fuzzer = Fuzzer(strategy=FuzzStrategy.EVOLUTIONARY)
            evo_inputs = evo_fuzzer.generate(
                base_input={"message": "Perform action"},
                count=30,
            )
            for i, inp in enumerate(evo_inputs):
                scenarios.append(Scenario(
                    name=f"evo_fuzz_{i}",
                    type=ScenarioType.ADVERSARIAL,
                    initial_entities=[{"type": "user", "name": "evo_user"}],
                    scripted_events=[{"time": 0, "type": "user_input", "data": inp}],
                    tags={"evolutionary_fuzz", "adversarial"},
                    difficulty=0.7,
                ))

            # Stress tests
            scenarios.extend(self.stress.generate_suite())

        self._generated_scenarios.extend(scenarios)

        logger.info(
            "adversarial_suite_generated",
            coverage=coverage,
            scenario_count=len(scenarios),
            edge_case_categories=list(self.edge_cases.coverage),
        )

        return scenarios

    # -- Targeted Generation --

    def generate_security_tests(self) -> List[Scenario]:
        """Generate security-focused test scenarios."""
        cases = self.edge_cases.discover_category("injection")
        return self.edge_cases.to_scenarios(cases)

    def generate_robustness_tests(self) -> List[Scenario]:
        """Generate robustness-focused test scenarios."""
        scenarios: List[Scenario] = []

        for cat in ["boundary", "encoding", "type_coercion"]:
            cases = self.edge_cases.discover_category(cat)
            scenarios.extend(self.edge_cases.to_scenarios(cases))

        fuzzed = self.fuzzer.generate(count=10)
        for i, inp in enumerate(fuzzed):
            scenarios.append(Scenario(
                name=f"robustness_fuzz_{i}",
                type=ScenarioType.ADVERSARIAL,
                scripted_events=[{"time": 0, "type": "user_input", "data": inp}],
                tags={"robustness"},
            ))

        return scenarios

    def generate_performance_tests(
        self,
        max_users: int = 100,
    ) -> List[Scenario]:
        """Generate performance-focused test scenarios."""
        self.stress.set_sla(
            max_latency_ms=2000.0,
            p99_latency_ms=1000.0,
            error_rate=0.05,
        )
        return self.stress.generate_suite()

    # -- Stats --

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_scenarios": len(self._generated_scenarios),
            "edge_case_coverage": list(self.edge_cases.coverage),
            "edge_cases_discovered": self.edge_cases.discovered_count,
            "fuzzer": self.fuzzer.stats,
        }

    def reset(self) -> None:
        self._generated_scenarios.clear()
        self.fuzzer.reset()
