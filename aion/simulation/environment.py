"""AION Simulation Environment - Main coordinator.

Orchestrates all simulation subsystems:
- World creation and management
- Scenario execution
- Agent sandboxing
- Timeline control (snapshots, branching, replay)
- Evaluation with statistical rigor
- Adversarial test generation
- Parallel scenario execution
"""

from __future__ import annotations

import asyncio
import copy
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.simulation.adversarial.generator import AdversarialGenerator
from aion.simulation.config import SimulationEnvironmentConfig, default_config
from aion.simulation.evaluation.evaluator import EvaluationResult, SimulationEvaluator
from aion.simulation.sandbox.agent_sandbox import AgentSandbox
from aion.simulation.scenarios.generator import ScenarioGenerator
from aion.simulation.timeline.manager import TimelineManager
from aion.simulation.types import (
    AgentInSimulation,
    Scenario,
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    WorldState,
)
from aion.simulation.world.engine import WorldEngine

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class SimulationEnvironment:
    """Main simulation environment coordinator.

    SOTA features:
    - Deterministic, reproducible simulation execution.
    - COW state snapshots with timeline branching.
    - Comprehensive evaluation with statistical A/B testing.
    - Adversarial test suite generation (fuzzing, edge cases, stress).
    - Parallel scenario execution.
    - Causal event graph for root-cause analysis.
    """

    def __init__(
        self,
        kernel: Optional["AIONKernel"] = None,
        env_config: Optional[SimulationEnvironmentConfig] = None,
    ) -> None:
        self.kernel = kernel
        self.env_config = env_config or default_config()

        # Current simulation state
        self._config: Optional[SimulationConfig] = None
        self._scenario: Optional[Scenario] = None
        self._world_engine: Optional[WorldEngine] = None
        self._agent_sandbox: Optional[AgentSandbox] = None
        self._timeline_manager: Optional[TimelineManager] = None

        # Components
        self.scenario_generator = ScenarioGenerator(kernel)
        self.evaluator = SimulationEvaluator()
        self.adversarial = AdversarialGenerator(
            seed=self.env_config.default_fuzz_seed,
        )

        # State
        self._status = SimulationStatus.CREATED
        self._result: Optional[SimulationResult] = None

        # History
        self._run_history: List[SimulationResult] = []

    # -- Simulation Lifecycle --

    async def create_simulation(
        self,
        scenario: Scenario,
        config: Optional[SimulationConfig] = None,
    ) -> str:
        """Create a new simulation.

        Returns:
            Simulation ID.
        """
        self._config = config or copy.deepcopy(self.env_config.default_sim_config)
        self._scenario = scenario

        # Initialize world engine
        self._world_engine = WorldEngine(self._config)
        await self._world_engine.initialize({
            "global": scenario.initial_state,
            "entities": scenario.initial_entities,
        })

        # Initialize sandbox
        self._agent_sandbox = AgentSandbox(self._world_engine, self.kernel)

        # Initialize timeline
        self._timeline_manager = TimelineManager(self._world_engine)
        self._timeline_manager.create_snapshot("initial_state")

        # Initialize result
        self._result = SimulationResult(
            scenario_id=scenario.id,
            status=SimulationStatus.INITIALIZING,
        )

        self._status = SimulationStatus.INITIALIZING

        logger.info(
            "simulation_created",
            scenario=scenario.name,
            sim_id=self._result.id,
        )

        return self._result.id

    async def load_agent(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
        tool_mocks: Optional[Dict[str, Any]] = None,
    ) -> AgentInSimulation:
        """Load an agent into the simulation."""
        if not self._agent_sandbox:
            raise RuntimeError("Simulation not created")

        return await self._agent_sandbox.load_agent(
            agent_id,
            config or {},
            tool_mocks,
        )

    async def run(self) -> SimulationResult:
        """Run the simulation to completion."""
        if not self._world_engine or not self._scenario:
            raise RuntimeError("Simulation not created")

        self._status = SimulationStatus.RUNNING
        self._result.status = SimulationStatus.RUNNING
        self._result.started_at = datetime.utcnow()

        wall_start = time.monotonic()

        logger.info("simulation_started", scenario=self._scenario.name)

        try:
            # Inject scripted events
            await self._inject_scripted_events()

            # Run simulation loop
            while self._should_continue():
                await self._step()

                # Periodic snapshots
                if (
                    self._config.record_state_snapshots
                    and self._world_engine.state.tick % self._config.snapshot_interval == 0
                ):
                    self._timeline_manager.create_snapshot()

            # Evaluate goals
            self._evaluate_goals()

            self._status = SimulationStatus.COMPLETED
            self._result.status = SimulationStatus.COMPLETED

        except Exception as exc:
            logger.error("simulation_failed", error=str(exc))
            self._status = SimulationStatus.FAILED
            self._result.status = SimulationStatus.FAILED
            self._result.errors.append(str(exc))

        # Finalize result
        wall_end = time.monotonic()
        self._result.completed_at = datetime.utcnow()
        self._result.total_ticks = self._world_engine.state.tick
        self._result.total_simulation_time = self._world_engine.state.simulation_time
        self._result.total_real_time = wall_end - wall_start
        self._result.final_state = self._world_engine.clone_state()
        self._result.event_count = len(self._world_engine.state.event_history)
        self._result.snapshot_count = self._timeline_manager.snapshots.count

        # Causal graph stats
        depth_stats = self._world_engine.causal_graph.depth_stats()
        self._result.causal_depth_max = depth_stats["max_depth"]
        self._result.causal_chains = depth_stats["root_count"]

        # Store metrics from world state
        self._result.metrics.update(self._world_engine.state.metrics)

        # Store in history
        self._run_history.append(self._result)

        logger.info(
            "simulation_completed",
            status=self._result.status.value,
            ticks=self._result.total_ticks,
            events=self._result.event_count,
            success=self._result.success,
            real_time=f"{self._result.total_real_time:.3f}s",
        )

        return self._result

    async def _step(self) -> None:
        """Execute one simulation step."""
        await self._world_engine.step()

    async def _inject_scripted_events(self) -> None:
        """Inject scripted events from scenario."""
        from aion.simulation.types import EventType, SimulationEvent

        for event_data in self._scenario.scripted_events:
            event = SimulationEvent(
                type=EventType(event_data.get("type", "user_input")),
                action=event_data.get("action", ""),
                data=event_data.get("data", {}),
                source_id=event_data.get("source"),
            )
            delay = int(event_data.get("time", 0) / self._config.tick_duration)
            self._world_engine.schedule_event(event, delay_ticks=max(0, delay))

    def _should_continue(self) -> bool:
        if self._status != SimulationStatus.RUNNING:
            return False
        if self._world_engine.state.tick >= self._config.max_ticks:
            return False
        if len(self._world_engine.state.event_history) >= self._config.max_events:
            return False
        if self._world_engine.state.simulation_time >= self._scenario.max_time:
            return False

        # Check failure criteria
        for criterion in self._scenario.failure_criteria:
            if self._evaluate_condition(criterion.get("condition", "")):
                return False

        # Check if all goals achieved early (skip terminal-only conditions)
        _TERMINAL_CONDITIONS = {"simulation_completed", "all_requests_processed"}
        dynamic_goals = [
            g for g in self._scenario.goals
            if g.get("condition", "") not in _TERMINAL_CONDITIONS
        ]
        if dynamic_goals:
            all_achieved = all(
                self._evaluate_condition(g.get("condition", ""))
                for g in dynamic_goals
            )
            if all_achieved:
                return False

        return True

    def _evaluate_goals(self) -> None:
        for goal in self._scenario.goals:
            if self._evaluate_condition(goal.get("condition", "")):
                self._result.goals_achieved.append(goal["name"])
            else:
                self._result.goals_failed.append(goal["name"])

        self._result.success = (
            len(self._result.goals_failed) == 0
            and (len(self._result.goals_achieved) > 0 or not self._scenario.goals)
        )

    def _evaluate_condition(self, condition: str) -> bool:
        state = self._world_engine.state

        if condition == "simulation_completed":
            return True
        elif condition == "response_sent":
            return any(e.action == "respond" for e in state.event_history)
        elif condition == "all_requests_processed":
            return True  # Simplified
        elif condition == "security_maintained":
            return not any(
                e.type.value == "error" and "injection" in (e.error or "").lower()
                for e in state.event_history
            )

        # Comparison operators
        for op in (">=", "<=", "!=", "==", ">", "<"):
            if op in condition:
                parts = condition.split(op, 1)
                try:
                    left = self._resolve_condition_value(parts[0].strip(), state)
                    right = self._resolve_condition_value(parts[1].strip(), state)
                    if op == ">=":
                        return float(left) >= float(right)
                    elif op == "<=":
                        return float(left) <= float(right)
                    elif op == "!=":
                        return left != right
                    elif op == "==":
                        return left == right
                    elif op == ">":
                        return float(left) > float(right)
                    elif op == "<":
                        return float(left) < float(right)
                except (ValueError, TypeError):
                    return False

        return False

    def _resolve_condition_value(self, expr: str, state: WorldState) -> Any:
        if expr in state.metrics:
            return state.metrics[expr]
        try:
            return float(expr)
        except ValueError:
            return expr

    # -- Control Methods --

    def pause(self) -> None:
        if self._world_engine:
            self._world_engine.pause()
        self._status = SimulationStatus.PAUSED

    def resume(self) -> None:
        if self._world_engine:
            self._world_engine.resume()
        self._status = SimulationStatus.RUNNING

    def stop(self) -> None:
        if self._world_engine:
            self._world_engine.stop()
        self._status = SimulationStatus.CANCELLED

    # -- Timeline Control --

    def create_snapshot(self, description: str = "") -> str:
        if not self._timeline_manager:
            raise RuntimeError("Simulation not created")
        snapshot = self._timeline_manager.create_snapshot(description)
        return snapshot.id

    def restore_snapshot(self, snapshot_id: str) -> bool:
        if not self._timeline_manager:
            return False
        return self._timeline_manager.restore_snapshot(snapshot_id)

    def create_branch(self, name: str, from_snapshot: Optional[str] = None) -> bool:
        if not self._timeline_manager:
            return False
        return self._timeline_manager.create_branch(name, from_snapshot)

    def switch_branch(self, name: str) -> bool:
        if not self._timeline_manager:
            return False
        return self._timeline_manager.switch_branch(name)

    def rewind(self, ticks: int) -> bool:
        if not self._timeline_manager:
            return False
        return self._timeline_manager.rewind(ticks)

    def compare_snapshots(self, id1: str, id2: str) -> Dict[str, Any]:
        if not self._timeline_manager:
            return {"error": "No timeline"}
        return self._timeline_manager.compare_snapshots(id1, id2)

    def compare_branches(self, b1: str, b2: str) -> Dict[str, Any]:
        if not self._timeline_manager:
            return {"error": "No timeline"}
        return self._timeline_manager.compare_branches(b1, b2)

    # -- Evaluation --

    async def evaluate(self) -> EvaluationResult:
        if not self._result or not self._world_engine:
            raise RuntimeError("No result to evaluate")
        return await self.evaluator.evaluate(
            self._result,
            self._world_engine.state,
        )

    # -- Batch Running --

    async def run_scenarios(
        self,
        scenarios: List[Scenario],
        config: Optional[SimulationConfig] = None,
    ) -> List[SimulationResult]:
        """Run multiple scenarios sequentially."""
        results: List[SimulationResult] = []
        for scenario in scenarios:
            await self.create_simulation(scenario, config)
            result = await self.run()
            results.append(result)
        return results

    async def run_scenarios_parallel(
        self,
        scenarios: List[Scenario],
        config: Optional[SimulationConfig] = None,
        max_parallel: Optional[int] = None,
    ) -> List[SimulationResult]:
        """Run multiple scenarios in parallel using semaphore limiting."""
        max_p = max_parallel or self.env_config.max_parallel_simulations
        semaphore = asyncio.Semaphore(max_p)
        results: List[SimulationResult] = [None] * len(scenarios)  # type: ignore

        async def _run_one(idx: int, scenario: Scenario) -> None:
            async with semaphore:
                env = SimulationEnvironment(
                    kernel=self.kernel,
                    env_config=self.env_config,
                )
                await env.create_simulation(scenario, config)
                results[idx] = await env.run()

        tasks = [_run_one(i, s) for i, s in enumerate(scenarios)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any exceptions
        return [r for r in results if isinstance(r, SimulationResult)]

    async def run_with_variations(
        self,
        scenario: Scenario,
        variations: List[Dict[str, Any]],
        config: Optional[SimulationConfig] = None,
    ) -> List[SimulationResult]:
        """Run scenario with different variations."""
        results: List[SimulationResult] = []
        for variation in variations:
            varied = self._apply_variation(scenario, variation)
            await self.create_simulation(varied, config)
            result = await self.run()
            result.metrics["variation"] = hash(str(variation))
            results.append(result)
        return results

    def _apply_variation(
        self,
        scenario: Scenario,
        variation: Dict[str, Any],
    ) -> Scenario:
        varied = copy.deepcopy(scenario)
        for key, value in variation.items():
            if hasattr(varied, key):
                setattr(varied, key, value)
            else:
                varied.config[key] = value
        return varied

    # -- A/B Testing --

    async def ab_test(
        self,
        scenario: Scenario,
        config_a: SimulationConfig,
        config_b: SimulationConfig,
        runs_per_group: int = 10,
    ) -> Dict[str, Any]:
        """Run A/B test between two configurations."""
        results_a: List[SimulationResult] = []
        results_b: List[SimulationResult] = []

        for _ in range(runs_per_group):
            await self.create_simulation(scenario, config_a)
            results_a.append(await self.run())

            await self.create_simulation(scenario, config_b)
            results_b.append(await self.run())

        return await self.evaluator.a_b_compare(results_a, results_b)

    # -- State Access --

    def get_status(self) -> SimulationStatus:
        return self._status

    def get_world_state(self) -> Optional[WorldState]:
        return self._world_engine.state if self._world_engine else None

    def get_result(self) -> Optional[SimulationResult]:
        return self._result

    def get_history(self) -> List[SimulationResult]:
        return list(self._run_history)

    @property
    def world_engine(self) -> Optional[WorldEngine]:
        return self._world_engine

    @property
    def agent_sandbox(self) -> Optional[AgentSandbox]:
        return self._agent_sandbox

    @property
    def timeline_manager(self) -> Optional[TimelineManager]:
        return self._timeline_manager
