"""AION World Engine - Core simulation world management.

The WorldEngine orchestrates all world subsystems:
- Entity management (ECS pattern)
- State management (COW snapshots, transactions)
- Event processing (causal DAG, priority bus)
- Rule evaluation (forward-chaining, conflict resolution)
- Constraint enforcement (invariants, resource limits)
- Deterministic execution (seeded RNG, ordered scheduling)
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import structlog

from aion.simulation.types import (
    Entity,
    EntityType,
    EventType,
    SimulationConfig,
    SimulationEvent,
    TimeMode,
    WorldState,
)
from aion.simulation.world.entities import EntityManager
from aion.simulation.world.events import CausalGraph, EventBus
from aion.simulation.world.physics import ConstraintSolver
from aion.simulation.world.rules import Rule, RulesEngine
from aion.simulation.world.state import WorldStateManager

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class WorldEngine:
    """Core engine for managing simulation worlds.

    SOTA features:
    - Deterministic execution with seeded RNG and monotonic event ordering.
    - Entity Component System for flexible entity composition.
    - Causal event graph for root-cause analysis and counterfactual reasoning.
    - Copy-on-write state snapshots for efficient timeline branching.
    - Forward-chaining rule engine with conflict resolution.
    - Constraint solver with automatic correction and propagation.
    - Transactional state changes with rollback support.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

        # Core subsystems
        self.state_manager = WorldStateManager()
        self.entity_manager = EntityManager()
        self.event_bus = EventBus()
        self.rules_engine = RulesEngine()
        self.constraint_solver = ConstraintSolver()
        self.causal_graph = CausalGraph()

        # Deterministic RNG
        self._rng = random.Random(config.seed)
        self._seed = config.seed

        # Running state
        self._running = False
        self._paused = False

        # Scheduled events (tick -> events)
        self._scheduled: Dict[int, List[SimulationEvent]] = {}

        # Behavior registry
        self._behaviors: Dict[str, Callable] = {}

        # Tick hooks
        self._pre_tick_hooks: List[Callable] = []
        self._post_tick_hooks: List[Callable] = []

        # Performance tracking
        self._total_events_processed: int = 0

    @property
    def state(self) -> WorldState:
        return self.state_manager.state

    @state.setter
    def state(self, new_state: WorldState) -> None:
        self.state_manager.state = new_state
        # Rebuild entity manager index from new state
        self.entity_manager.load_entities(new_state.entities)

    # -- Initialization --

    async def initialize(self, initial_state: Dict[str, Any]) -> None:
        """Initialize the world with initial state."""
        state = self.state_manager.state

        # Set global state
        state.global_state = initial_state.get("global", {})

        # Capture RNG state for deterministic replay
        if self.config.deterministic:
            state.rng_state = self._rng.getstate()

        # Create initial entities
        for entity_data in initial_state.get("entities", []):
            entity = self.entity_manager.create_from_data(entity_data)
            state.add_entity(entity)

        # Reset time
        state.simulation_time = 0.0
        state.tick = 0
        state.event_sequence = 0

        # Commit initial state
        self.state_manager.commit()

        logger.info(
            "world_initialized",
            entity_count=self.entity_manager.count,
            seed=self._seed,
            deterministic=self.config.deterministic,
        )

    # -- Tick Execution --

    async def step(self) -> List[SimulationEvent]:
        """Advance the simulation by one tick.

        Execution order (deterministic):
        1. Pre-tick hooks
        2. Process scheduled events for this tick
        3. Process pending events
        4. Run entity behaviors
        5. Evaluate rules (with chaining)
        6. Enforce constraints (with propagation)
        7. Emit tick event
        8. Post-tick hooks

        Returns:
            All events produced during this tick.
        """
        if self._paused:
            return []

        state = self.state_manager.state
        events: List[SimulationEvent] = []

        # Begin transaction for rollback support
        self.state_manager.begin_transaction()

        # Increment tick
        state.tick += 1
        state.simulation_time += self.config.tick_duration

        # 1. Pre-tick hooks
        for hook in self._pre_tick_hooks:
            try:
                result = hook(state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("pre_tick_hook_error", error=str(exc))

        # 2. Process scheduled events
        scheduled = self._scheduled.pop(state.tick, [])
        for event in scheduled:
            event.tick = state.tick
            event.sequence_number = state.next_event_sequence()
            result_events = await self._process_event(event)
            events.append(event)
            events.extend(result_events)

        # 3. Process pending events
        pending = state.pending_events.copy()
        state.pending_events.clear()
        for event in pending:
            event.tick = state.tick
            event.sequence_number = state.next_event_sequence()
            result_events = await self._process_event(event)
            events.append(event)
            events.extend(result_events)

        # 4. Run entity behaviors
        behavior_events = await self._run_behaviors()
        events.extend(behavior_events)

        # 5. Evaluate rules
        rule_events = await self.rules_engine.evaluate_chain(state)
        events.extend(rule_events)

        # 6. Enforce constraints
        violations = await self.constraint_solver.check_all(state)
        for violation in violations:
            if not violation.corrected and violation.severity == "error":
                events.append(SimulationEvent(
                    type=EventType.CONSTRAINT_VIOLATION,
                    action=violation.constraint_name,
                    data={"message": violation.message, "corrected": violation.corrected},
                    simulation_time=state.simulation_time,
                    tick=state.tick,
                    success=False,
                    error=violation.message,
                    sequence_number=state.next_event_sequence(),
                ))

        # 7. Tick event
        tick_event = SimulationEvent(
            type=EventType.TIME_TICK,
            action="tick",
            simulation_time=state.simulation_time,
            tick=state.tick,
            data={"tick": state.tick, "event_count": len(events)},
            sequence_number=state.next_event_sequence(),
        )
        events.append(tick_event)

        # 8. Record events
        if self.config.record_all_events:
            state.event_history.extend(events)

        # Track causal graph
        if self.config.record_causal_graph:
            for event in events:
                self.causal_graph.add_event(event, caused_by=event.caused_by)

        # 9. Post-tick hooks
        for hook in self._post_tick_hooks:
            try:
                result = hook(state, events)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("post_tick_hook_error", error=str(exc))

        # Commit transaction
        self.state_manager.commit()
        self._total_events_processed += len(events)

        return events

    async def run(self, max_ticks: Optional[int] = None) -> None:
        """Run the simulation until completion or max ticks."""
        max_ticks = max_ticks or self.config.max_ticks
        self._running = True

        while self._running and self.state.tick < max_ticks:
            if self._paused:
                await asyncio.sleep(0.01)
                continue

            await self.step()

            # Check event limit
            if len(self.state.event_history) >= self.config.max_events:
                logger.warning("max_events_reached", count=len(self.state.event_history))
                break

            # Time mode sleep
            if self.config.time_mode == TimeMode.REAL_TIME:
                await asyncio.sleep(self.config.tick_duration)
            elif self.config.time_mode == TimeMode.ACCELERATED:
                await asyncio.sleep(self.config.tick_duration / self.config.time_scale)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False

    # -- Entity Management --

    def create_entity(
        self,
        entity_type: EntityType,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        components: Optional[Dict[str, Dict[str, Any]]] = None,
        behaviors: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
    ) -> Entity:
        """Create a new entity in the world."""
        entity = self.entity_manager.create(
            entity_type=entity_type,
            name=name,
            properties=properties,
            components=components,
            behaviors=behaviors,
            tags=tags,
        )
        self.state.add_entity(entity)
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.state.get_entity(entity_id)

    def query_entities(
        self,
        entity_type: Optional[EntityType] = None,
        tags: Optional[Set[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Query entities matching criteria."""
        return self.entity_manager.query(
            entity_type=entity_type,
            tags=tags,
            properties=properties,
        )

    # -- Event Management --

    def emit_event(self, event: SimulationEvent) -> None:
        """Emit an event to be processed in the current or next tick."""
        event.sequence_number = self.state.next_event_sequence()
        self.state.pending_events.append(event)

    def schedule_event(self, event: SimulationEvent, delay_ticks: int = 0) -> None:
        """Schedule an event for a future tick."""
        target_tick = self.state.tick + delay_ticks
        event.simulation_time = self.state.simulation_time + (
            delay_ticks * self.config.tick_duration
        )
        if target_tick not in self._scheduled:
            self._scheduled[target_tick] = []
        self._scheduled[target_tick].append(event)

    async def _process_event(self, event: SimulationEvent) -> List[SimulationEvent]:
        """Process a single event through the event bus."""
        return await self.event_bus.emit(event)

    def on_event(self, event_type: EventType, handler: Callable, priority: int = 100) -> None:
        """Register an event handler."""
        self.event_bus.on(event_type, handler, priority=priority)

    # -- Behaviors --

    def register_behavior(self, name: str, behavior: Callable) -> None:
        """Register an entity behavior."""
        self._behaviors[name] = behavior

    async def _run_behaviors(self) -> List[SimulationEvent]:
        """Run behaviors for all entities."""
        events: List[SimulationEvent] = []
        state = self.state

        for entity in list(state.entities.values()):
            for behavior_name in entity.behaviors:
                behavior = self._behaviors.get(behavior_name)
                if behavior is None:
                    continue
                try:
                    result = behavior(entity, state)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, SimulationEvent):
                        result.sequence_number = state.next_event_sequence()
                        events.append(result)
                    elif isinstance(result, list):
                        for evt in result:
                            if isinstance(evt, SimulationEvent):
                                evt.sequence_number = state.next_event_sequence()
                                events.append(evt)
                except Exception as exc:
                    logger.error(
                        "behavior_error",
                        behavior=behavior_name,
                        entity_id=entity.id,
                        error=str(exc),
                    )

        return events

    # -- Rules --

    def add_rule(self, rule: Rule) -> None:
        self.rules_engine.add_rule(rule)

    # -- Tick Hooks --

    def add_pre_tick_hook(self, hook: Callable) -> None:
        self._pre_tick_hooks.append(hook)

    def add_post_tick_hook(self, hook: Callable) -> None:
        self._post_tick_hooks.append(hook)

    # -- State Access --

    def get_state(self) -> WorldState:
        return self.state

    def set_state(self, state: WorldState) -> None:
        self.state = state

    def clone_state(self) -> WorldState:
        return self.state_manager.snapshot()

    # -- Deterministic RNG --

    def random(self) -> float:
        """Get next deterministic random value."""
        return self._rng.random()

    def random_int(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def random_choice(self, seq: list) -> Any:
        return self._rng.choice(seq)

    def capture_rng_state(self) -> Any:
        return self._rng.getstate()

    def restore_rng_state(self, rng_state: Any) -> None:
        self._rng.setstate(rng_state)

    # -- Stats --

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "tick": self.state.tick,
            "simulation_time": self.state.simulation_time,
            "entity_count": self.entity_manager.count,
            "total_events_processed": self._total_events_processed,
            "event_history_size": len(self.state.event_history),
            "causal_graph_size": self.causal_graph.size,
            "rules": self.rules_engine.stats,
            "constraints": self.constraint_solver.violation_summary(),
            "running": self._running,
            "paused": self._paused,
        }
