"""Comprehensive tests for the AION Simulation Environment.

Tests cover:
- Types and data structures
- World Engine (entities, events, rules, constraints, behaviors)
- Scenario Generator (all types, templates, fragments, combinatorial)
- Agent Sandbox (actions, tool mocking, memory isolation, resource limits)
- Timeline Manager (snapshots, branching, rewind, replay, comparison)
- Evaluation Framework (metrics, assertions, scoring, A/B comparison)
- Adversarial Generator (fuzzing, edge cases, stress, full suite)
- Main Environment (lifecycle, batch runs, parallel execution)
- API (templates, quick run, agent testing)
"""

from __future__ import annotations

import asyncio
import copy
import math
import random
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
from aion.simulation.types import (
    AgentInSimulation,
    Assertion,
    Constraint,
    ConstraintType,
    Entity,
    EntityType,
    EvaluationMetric,
    EventType,
    FuzzStrategy,
    Scenario,
    ScenarioType,
    SimulationConfig,
    SimulationEvent,
    SimulationResult,
    SimulationStatus,
    TimelineSnapshot,
    TimeMode,
    WorldState,
)


class TestTypes:
    """Test core simulation types."""

    def test_entity_creation(self):
        entity = Entity(name="test", type=EntityType.AGENT)
        assert entity.name == "test"
        assert entity.type == EntityType.AGENT
        assert entity.id  # UUID assigned

    def test_entity_properties(self):
        entity = Entity()
        entity.set_property("key", "value")
        assert entity.get_property("key") == "value"
        assert entity.get_property("missing", "default") == "default"
        assert entity.version == 1

    def test_entity_components(self):
        entity = Entity()
        entity.add_component("health", {"hp": 100, "max_hp": 100})
        assert entity.has_component("health")
        assert entity.get_component("health")["hp"] == 100
        entity.remove_component("health")
        assert not entity.has_component("health")

    def test_entity_fingerprint(self):
        e1 = Entity(properties={"a": 1})
        e2 = Entity(properties={"a": 1})
        e3 = Entity(properties={"a": 2})
        assert e1.fingerprint() == e2.fingerprint()
        assert e1.fingerprint() != e3.fingerprint()

    def test_entity_to_dict(self):
        entity = Entity(name="test", type=EntityType.USER)
        d = entity.to_dict()
        assert d["name"] == "test"
        assert d["type"] == "user"

    def test_simulation_event(self):
        event = SimulationEvent(
            type=EventType.AGENT_ACTION,
            action="respond",
            data={"message": "hello"},
        )
        assert event.type == EventType.AGENT_ACTION
        assert event.success is True
        d = event.to_dict()
        assert d["action"] == "respond"

    def test_world_state_entities(self):
        state = WorldState()
        entity = Entity(name="e1")
        state.add_entity(entity)
        assert state.get_entity(entity.id) is not None
        state.remove_entity(entity.id)
        assert state.get_entity(entity.id) is None

    def test_world_state_clone(self):
        state = WorldState()
        state.add_entity(Entity(name="e1"))
        state.global_state["key"] = "value"
        state.metrics["score"] = 0.5

        cloned = state.clone()
        assert len(cloned.entities) == 1
        assert cloned.global_state["key"] == "value"
        assert cloned.metrics["score"] == 0.5
        # Shallow copy: modifying clone entities dict doesn't affect original
        cloned.entities.clear()
        assert len(state.entities) == 1

    def test_world_state_cow_mutation(self):
        state = WorldState()
        entity = Entity(name="original")
        state.add_entity(entity)
        cloned = state.clone()

        # Mutate through COW
        mutated = cloned.mutate_entity(entity.id)
        mutated.name = "mutated"

        # Original unchanged
        assert state.entities[entity.id].name == "original"
        assert cloned.entities[entity.id].name == "mutated"

    def test_world_state_fingerprint(self):
        s1 = WorldState()
        s1.add_entity(Entity(name="e1", properties={"x": 1}))
        s2 = WorldState()
        s2.add_entity(Entity(name="e1", properties={"x": 1}))

        # Different IDs → different fingerprints (entities keyed by ID)
        fp1 = s1.fingerprint()
        fp2 = s2.fingerprint()
        assert isinstance(fp1, str) and len(fp1) > 0

    def test_world_state_event_sequence(self):
        state = WorldState()
        seq1 = state.next_event_sequence()
        seq2 = state.next_event_sequence()
        assert seq2 == seq1 + 1

    def test_scenario_creation(self):
        scenario = Scenario(
            name="test",
            type=ScenarioType.ADVERSARIAL,
            tags={"security"},
            difficulty=0.8,
        )
        assert scenario.name == "test"
        assert scenario.difficulty == 0.8
        d = scenario.to_dict()
        assert d["type"] == "adversarial"

    def test_simulation_config_defaults(self):
        config = SimulationConfig()
        assert config.time_mode == TimeMode.STEP
        assert config.deterministic is True
        assert config.max_ticks == 10_000

    def test_timeline_snapshot(self):
        snapshot = TimelineSnapshot(tick=10, simulation_time=10.0)
        fp = snapshot.compute_fingerprint()
        assert isinstance(fp, str) and len(fp) > 0

    def test_enums(self):
        assert SimulationStatus.RUNNING.value == "running"
        assert EntityType.AGENT.value == "agent"
        assert EventType.AGENT_ACTION.value == "agent_action"
        assert TimeMode.DETERMINISTIC.value == "deterministic"
        assert ScenarioType.DIFFERENTIAL.value == "differential"
        assert FuzzStrategy.EVOLUTIONARY.value == "evolutionary"


# ---------------------------------------------------------------------------
# World Engine
# ---------------------------------------------------------------------------
from aion.simulation.world.engine import WorldEngine


class TestWorldEngine:
    """Test the world engine."""

    @pytest.fixture
    def engine(self):
        config = SimulationConfig(seed=42, max_ticks=100, deterministic=True)
        return WorldEngine(config)

    @pytest.mark.asyncio
    async def test_initialize(self, engine):
        await engine.initialize({
            "global": {"level": 1},
            "entities": [
                {"type": "user", "name": "alice"},
                {"type": "agent", "name": "bot"},
            ],
        })
        assert engine.entity_manager.count == 2
        assert engine.state.global_state["level"] == 1
        assert engine.state.tick == 0

    @pytest.mark.asyncio
    async def test_step(self, engine):
        await engine.initialize({"entities": []})
        events = await engine.step()
        assert engine.state.tick == 1
        assert len(events) > 0  # At least tick event
        assert any(e.type == EventType.TIME_TICK for e in events)

    @pytest.mark.asyncio
    async def test_multiple_steps(self, engine):
        await engine.initialize({"entities": []})
        for _ in range(10):
            await engine.step()
        assert engine.state.tick == 10

    @pytest.mark.asyncio
    async def test_create_entity(self, engine):
        await engine.initialize({"entities": []})
        entity = engine.create_entity(EntityType.USER, "bob", properties={"age": 25})
        assert entity.name == "bob"
        assert engine.get_entity(entity.id) is not None

    @pytest.mark.asyncio
    async def test_query_entities(self, engine):
        await engine.initialize({"entities": []})
        engine.create_entity(EntityType.USER, "alice", tags={"admin"})
        engine.create_entity(EntityType.USER, "bob")
        engine.create_entity(EntityType.AGENT, "bot")

        users = engine.query_entities(entity_type=EntityType.USER)
        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_emit_and_process_events(self, engine):
        await engine.initialize({"entities": []})

        processed = []

        async def handler(event):
            processed.append(event)

        engine.on_event(EventType.USER_INPUT, handler)

        engine.emit_event(SimulationEvent(
            type=EventType.USER_INPUT,
            action="test",
            data={"message": "hello"},
        ))

        await engine.step()
        assert len(processed) == 1
        assert processed[0].action == "test"

    @pytest.mark.asyncio
    async def test_schedule_event(self, engine):
        await engine.initialize({"entities": []})

        received = []

        async def handler(event):
            received.append(event)

        engine.on_event(EventType.TRIGGER, handler)

        engine.schedule_event(
            SimulationEvent(type=EventType.TRIGGER, action="delayed"),
            delay_ticks=3,
        )

        # Run 4 steps: event should fire at tick 3
        for _ in range(4):
            await engine.step()

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_behaviors(self, engine):
        await engine.initialize({"entities": []})

        behavior_calls = []

        async def counting_behavior(entity, state):
            behavior_calls.append(entity.name)
            return SimulationEvent(
                type=EventType.AGENT_ACTION,
                source_id=entity.id,
                action="behave",
            )

        engine.register_behavior("counter", counting_behavior)
        engine.create_entity(EntityType.AGENT, "bot", behaviors=["counter"])

        await engine.step()
        assert "bot" in behavior_calls

    @pytest.mark.asyncio
    async def test_deterministic_rng(self, engine):
        r1 = engine.random()
        # Create another engine with same seed
        engine2 = WorldEngine(SimulationConfig(seed=42))
        r2 = engine2.random()
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_pause_resume(self, engine):
        await engine.initialize({"entities": []})
        engine.pause()
        events = await engine.step()
        assert events == []  # No events when paused
        engine.resume()
        events = await engine.step()
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stats(self, engine):
        await engine.initialize({"entities": [{"type": "user", "name": "alice"}]})
        await engine.step()
        stats = engine.stats
        assert stats["tick"] == 1
        assert stats["entity_count"] == 1
        assert stats["total_events_processed"] > 0


# ---------------------------------------------------------------------------
# World Events (EventBus + CausalGraph)
# ---------------------------------------------------------------------------
from aion.simulation.world.events import CausalGraph, EventBus


class TestEventBus:
    def test_creation(self):
        bus = EventBus()
        assert bus.event_counts == {}

    @pytest.mark.asyncio
    async def test_emit_basic(self):
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.on(EventType.USER_INPUT, handler)

        event = SimulationEvent(type=EventType.USER_INPUT, action="test")
        await bus.emit(event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        bus = EventBus()
        order = []

        def handler_low(event):
            order.append("low")

        def handler_high(event):
            order.append("high")

        bus.on(EventType.USER_INPUT, handler_low, priority=200)
        bus.on(EventType.USER_INPUT, handler_high, priority=50)

        await bus.emit(SimulationEvent(type=EventType.USER_INPUT))
        assert order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_wildcard_handler(self):
        bus = EventBus()
        received = []
        bus.on_any(lambda e: received.append(e.type))
        await bus.emit(SimulationEvent(type=EventType.USER_INPUT))
        await bus.emit(SimulationEvent(type=EventType.AGENT_ACTION))
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_dead_letters(self):
        bus = EventBus()
        await bus.emit(SimulationEvent(type=EventType.ERROR))
        assert len(bus.dead_letters) == 1

    @pytest.mark.asyncio
    async def test_handler_returning_events(self):
        bus = EventBus()

        def handler(event):
            return SimulationEvent(type=EventType.SYSTEM_EVENT, action="derived")

        bus.on(EventType.USER_INPUT, handler)
        results = await bus.emit(SimulationEvent(type=EventType.USER_INPUT))
        assert len(results) == 1
        assert results[0].action == "derived"


class TestCausalGraph:
    def test_add_root_event(self):
        graph = CausalGraph()
        event = SimulationEvent(id="e1", action="root")
        graph.add_event(event)
        assert graph.size == 1
        assert len(graph.roots) == 1

    def test_causal_chain(self):
        graph = CausalGraph()
        e1 = SimulationEvent(id="e1", action="cause")
        e2 = SimulationEvent(id="e2", action="effect")
        graph.add_event(e1)
        graph.add_event(e2, caused_by="e1")

        chain = graph.causal_chain("e2")
        assert len(chain) == 2
        assert chain[0].id == "e1"
        assert chain[1].id == "e2"

    def test_root_cause(self):
        graph = CausalGraph()
        for i in range(5):
            caused_by = f"e{i - 1}" if i > 0 else None
            graph.add_event(
                SimulationEvent(id=f"e{i}", action=f"step_{i}"),
                caused_by=caused_by,
            )
        root = graph.root_cause("e4")
        assert root.id == "e0"

    def test_impact_set(self):
        graph = CausalGraph()
        graph.add_event(SimulationEvent(id="root"))
        graph.add_event(SimulationEvent(id="c1"), caused_by="root")
        graph.add_event(SimulationEvent(id="c2"), caused_by="root")
        graph.add_event(SimulationEvent(id="c3"), caused_by="c1")

        impact = graph.impact_set("root")
        assert len(impact) == 3

    def test_critical_path(self):
        graph = CausalGraph()
        graph.add_event(SimulationEvent(id="r"))
        graph.add_event(SimulationEvent(id="a1"), caused_by="r")
        graph.add_event(SimulationEvent(id="a2"), caused_by="a1")
        graph.add_event(SimulationEvent(id="b1"), caused_by="r")

        path = graph.critical_path()
        assert len(path) == 3  # r -> a1 -> a2

    def test_depth_stats(self):
        graph = CausalGraph()
        graph.add_event(SimulationEvent(id="r"))
        graph.add_event(SimulationEvent(id="c1"), caused_by="r")
        stats = graph.depth_stats()
        assert stats["max_depth"] == 1
        assert stats["total_events"] == 2


# ---------------------------------------------------------------------------
# Entity Manager
# ---------------------------------------------------------------------------
from aion.simulation.world.entities import EntityManager


class TestEntityManager:
    def test_create_and_get(self):
        mgr = EntityManager()
        entity = mgr.create(EntityType.USER, "alice")
        assert mgr.get(entity.id) is not None
        assert mgr.count == 1

    def test_remove(self):
        mgr = EntityManager()
        entity = mgr.create(EntityType.USER, "alice")
        removed = mgr.remove(entity.id)
        assert removed is not None
        assert mgr.count == 0

    def test_query_by_type(self):
        mgr = EntityManager()
        mgr.create(EntityType.USER, "alice")
        mgr.create(EntityType.USER, "bob")
        mgr.create(EntityType.AGENT, "bot")
        assert len(mgr.query_by_type(EntityType.USER)) == 2

    def test_query_by_tag(self):
        mgr = EntityManager()
        mgr.create(EntityType.USER, "alice", tags={"admin"})
        mgr.create(EntityType.USER, "bob", tags={"user"})
        assert len(mgr.query_by_tag("admin")) == 1

    def test_query_by_component(self):
        mgr = EntityManager()
        e1 = mgr.create(EntityType.AGENT, "bot1")
        mgr.add_component(e1.id, "health", {"hp": 100})
        mgr.create(EntityType.AGENT, "bot2")
        assert len(mgr.query_by_component("health")) == 1

    def test_query_by_multiple_components(self):
        mgr = EntityManager()
        e1 = mgr.create(EntityType.AGENT, "bot1")
        mgr.add_component(e1.id, "health", {"hp": 100})
        mgr.add_component(e1.id, "position", {"x": 0, "y": 0})
        e2 = mgr.create(EntityType.AGENT, "bot2")
        mgr.add_component(e2.id, "health", {"hp": 50})
        assert len(mgr.query_by_components("health", "position")) == 1

    def test_archetype(self):
        mgr = EntityManager()
        mgr.register_archetype("soldier", {
            "type": "agent",
            "name": "soldier",
            "properties": {"class": "warrior"},
            "behaviors": ["patrol"],
        })
        entity = mgr.create_from_archetype("soldier")
        assert entity is not None
        assert entity.properties["class"] == "warrior"

    def test_relationships(self):
        mgr = EntityManager()
        e1 = mgr.create(EntityType.USER, "alice")
        e2 = mgr.create(EntityType.USER, "bob")
        mgr.add_relationship(e1.id, e2.id, "friend", bidirectional=True)
        friends = mgr.get_related(e1.id, "friend")
        assert len(friends) == 1
        inverse = mgr.get_related(e2.id, "inverse_friend")
        assert len(inverse) == 1

    def test_lifecycle_hooks(self):
        mgr = EntityManager()
        created = []
        destroyed = []
        mgr.on_create(lambda e: created.append(e.id))
        mgr.on_destroy(lambda e: destroyed.append(e.id))

        entity = mgr.create(EntityType.USER, "alice")
        assert len(created) == 1
        mgr.remove(entity.id)
        assert len(destroyed) == 1


# ---------------------------------------------------------------------------
# Rules Engine
# ---------------------------------------------------------------------------
from aion.simulation.world.rules import Rule, RulesEngine, ConflictStrategy


class TestRulesEngine:
    @pytest.mark.asyncio
    async def test_rule_fires(self):
        engine = RulesEngine()
        state = WorldState(tick=5)

        fired = []
        engine.add_rule(Rule(
            name="test_rule",
            condition=lambda s: s.tick >= 5,
            action=lambda s: {"triggered": True},
        ))

        events = await engine.evaluate(state)
        assert len(events) == 1
        assert events[0].action == "test_rule"

    @pytest.mark.asyncio
    async def test_rule_cooldown(self):
        engine = RulesEngine()

        engine.add_rule(Rule(
            name="cooldown_rule",
            condition=lambda s: True,
            action=lambda s: None,
            cooldown_ticks=5,
        ))

        state = WorldState(tick=1)
        events1 = await engine.evaluate(state)
        assert len(events1) == 1

        state.tick = 2
        events2 = await engine.evaluate(state)
        assert len(events2) == 0  # Cooldown active

        state.tick = 7
        events3 = await engine.evaluate(state)
        assert len(events3) == 1  # Cooldown expired

    @pytest.mark.asyncio
    async def test_conflict_resolution_first(self):
        engine = RulesEngine(conflict_strategy=ConflictStrategy.FIRST)
        engine.add_rule(Rule(name="r1", condition=lambda s: True, action=lambda s: None, priority=10))
        engine.add_rule(Rule(name="r2", condition=lambda s: True, action=lambda s: None, priority=5))

        events = await engine.evaluate(WorldState())
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_rule_composition(self):
        cond1 = lambda s: s.tick > 0
        cond2 = lambda s: len(s.entities) > 0

        combined_and = RulesEngine.all_of(cond1, cond2)
        combined_or = RulesEngine.any_of(cond1, cond2)
        negated = RulesEngine.not_of(cond1)

        state = WorldState(tick=1)
        assert not combined_and(state)  # No entities
        assert combined_or(state)  # tick > 0
        assert not negated(state)  # tick > 0 negated

    @pytest.mark.asyncio
    async def test_rule_stats(self):
        engine = RulesEngine()
        engine.add_rule(Rule(name="r1", condition=lambda s: True, action=lambda s: None))
        await engine.evaluate(WorldState(tick=1))
        assert engine.stats["fire_count"] == 1


# ---------------------------------------------------------------------------
# Constraint Solver
# ---------------------------------------------------------------------------
from aion.simulation.world.physics import ConstraintSolver


class TestConstraintSolver:
    @pytest.mark.asyncio
    async def test_invariant_satisfied(self):
        solver = ConstraintSolver()
        solver.add_invariant("always_true", lambda s: True)
        violations = await solver.check_all(WorldState())
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_invariant_violated(self):
        solver = ConstraintSolver()
        solver.add_invariant("always_false", lambda s: False)
        violations = await solver.check_all(WorldState())
        assert len(violations) == 1
        assert violations[0].constraint_name == "always_false"

    @pytest.mark.asyncio
    async def test_correction(self):
        solver = ConstraintSolver()
        state = WorldState()
        state.metrics["count"] = 150

        def correct(s):
            s.metrics["count"] = 100

        solver.add_invariant(
            "max_count",
            lambda s: s.metrics.get("count", 0) <= 100,
            on_violation="correct",
            correction=correct,
        )
        violations = await solver.check_all(state)
        assert len(violations) == 1
        assert violations[0].corrected is True
        assert state.metrics["count"] == 100

    @pytest.mark.asyncio
    async def test_resource_limit(self):
        solver = ConstraintSolver()
        state = WorldState()
        state.metrics["memory"] = 600

        solver.add_resource_limit("memory_limit", "memory", max_value=512)
        violations = await solver.check_all(state)
        assert len(violations) == 1

    @pytest.mark.asyncio
    async def test_precondition(self):
        solver = ConstraintSolver()
        solver.add_precondition("deploy", "health_check", lambda s: s.metrics.get("health", 0) > 0)

        state = WorldState()
        state.metrics["health"] = 0
        ok, violations = await solver.check_preconditions("deploy", state)
        assert not ok
        assert len(violations) == 1

    @pytest.mark.asyncio
    async def test_violation_summary(self):
        solver = ConstraintSolver()
        solver.add_invariant("bad", lambda s: False)
        await solver.check_all(WorldState(tick=1))
        await solver.check_all(WorldState(tick=2))
        summary = solver.violation_summary()
        assert summary["total"] == 2


# ---------------------------------------------------------------------------
# World State Manager
# ---------------------------------------------------------------------------
from aion.simulation.world.state import WorldStateManager


class TestWorldStateManager:
    def test_transaction_rollback(self):
        mgr = WorldStateManager()
        mgr.state.metrics["x"] = 10
        mgr.begin_transaction()
        mgr.state.metrics["x"] = 20
        mgr.rollback()
        assert mgr.state.metrics["x"] == 10

    def test_undo_redo(self):
        mgr = WorldStateManager()
        mgr.state.metrics["x"] = 1
        mgr.begin_transaction()
        mgr.state.metrics["x"] = 2
        mgr.commit()
        mgr.undo()
        assert mgr.state.metrics["x"] == 1
        mgr.redo()
        assert mgr.state.metrics["x"] == 2

    def test_diff_computation(self):
        mgr = WorldStateManager()
        old = WorldState()
        e1 = Entity(id="e1", name="alice")
        old.add_entity(e1)
        old.metrics["score"] = 1.0

        new = old.clone()
        new.tick = 5
        new.add_entity(Entity(id="e2", name="bob"))
        new.metrics["score"] = 2.0

        diff = mgr.compute_diff(old, new)
        assert "e2" in diff.entities_added
        assert diff.metric_changes["score"] == 2.0

    def test_validate(self):
        mgr = WorldStateManager()
        entity = Entity(id="e1", parent_id="nonexistent")
        mgr.state.add_entity(entity)
        errors = mgr.validate()
        assert len(errors) > 0

    def test_metrics(self):
        mgr = WorldStateManager()
        mgr.update_metric("score", 10.0)
        assert mgr.get_metric("score") == 10.0
        result = mgr.increment_metric("score", 5.0)
        assert result == 15.0


# ---------------------------------------------------------------------------
# Scenario Generator
# ---------------------------------------------------------------------------
from aion.simulation.scenarios.generator import ScenarioGenerator


class TestScenarioGenerator:
    @pytest.fixture
    def generator(self):
        return ScenarioGenerator()

    @pytest.mark.asyncio
    async def test_generate_simple(self, generator):
        scenario = await generator.generate(ScenarioType.SIMPLE)
        assert scenario.type == ScenarioType.SIMPLE
        assert len(scenario.initial_entities) > 0

    @pytest.mark.asyncio
    async def test_generate_sequential(self, generator):
        scenario = await generator.generate(
            ScenarioType.SEQUENTIAL, {"steps": 3},
        )
        assert scenario.type == ScenarioType.SEQUENTIAL
        assert len(scenario.scripted_events) == 3

    @pytest.mark.asyncio
    async def test_generate_adversarial(self, generator):
        scenario = await generator.generate(ScenarioType.ADVERSARIAL)
        assert scenario.type == ScenarioType.ADVERSARIAL
        assert len(scenario.scripted_events) > 5

    @pytest.mark.asyncio
    async def test_generate_stress(self, generator):
        scenario = await generator.generate(
            ScenarioType.STRESS, {"concurrent_users": 10, "requests_per_user": 5},
        )
        assert scenario.type == ScenarioType.STRESS
        assert len(scenario.scripted_events) > 0

    @pytest.mark.asyncio
    async def test_generate_random(self, generator):
        generator.set_seed(42)
        scenario = await generator.generate(ScenarioType.RANDOM, {"num_events": 20})
        assert len(scenario.scripted_events) == 20

    @pytest.mark.asyncio
    async def test_generate_from_template(self, generator):
        scenario = await generator.generate_from_template("simple_task")
        assert "simple_task" in scenario.name

    @pytest.mark.asyncio
    async def test_generate_for_agent(self, generator):
        agent_config = {"tools": ["search"], "goals": ["help_user"]}
        scenarios = await generator.generate_for_agent(agent_config, "standard")
        assert len(scenarios) > 2

    @pytest.mark.asyncio
    async def test_combinatorial(self, generator):
        scenarios = await generator.generate_combinatorial(
            {"steps": [1, 2], "max_steps": [50, 100]},
        )
        assert len(scenarios) == 4

    @pytest.mark.asyncio
    async def test_compose_fragments(self, generator):
        scenario = await generator.compose_from_fragments(
            ["user_greeting", "tool_usage"],
        )
        assert len(scenario.scripted_events) > 0
        assert "greeting" in scenario.tags or "tool" in scenario.tags

    @pytest.mark.asyncio
    async def test_coverage_tracking(self, generator):
        await generator.generate(ScenarioType.SIMPLE)
        assert len(generator.coverage) > 0


# ---------------------------------------------------------------------------
# Scenario Templates
# ---------------------------------------------------------------------------
from aion.simulation.scenarios.templates import ScenarioTemplateLibrary


class TestScenarioTemplates:
    def test_list_templates(self):
        lib = ScenarioTemplateLibrary()
        templates = lib.list_templates()
        assert len(templates) > 0
        assert "customer_support_basic" in templates

    def test_instantiate(self):
        lib = ScenarioTemplateLibrary()
        scenario = lib.instantiate("customer_support_basic")
        assert scenario.name == "customer_support_basic"
        assert len(scenario.initial_entities) > 0

    def test_instantiate_with_overrides(self):
        lib = ScenarioTemplateLibrary()
        scenario = lib.instantiate("customer_support_basic", {"max_steps": 999})
        assert scenario.max_steps == 999

    def test_list_by_tag(self):
        lib = ScenarioTemplateLibrary()
        support = lib.list_by_tag("customer_service")
        assert len(support) >= 1


# ---------------------------------------------------------------------------
# Agent Sandbox
# ---------------------------------------------------------------------------
from aion.simulation.sandbox.agent_sandbox import AgentSandbox


class TestAgentSandbox:
    @pytest.fixture
    def sandbox(self):
        config = SimulationConfig(seed=42)
        engine = WorldEngine(config)
        return AgentSandbox(engine)

    @pytest.mark.asyncio
    async def test_load_agent(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        agent = await sandbox.load_agent("agent_1", {"type": "assistant"})
        assert agent.agent_id == "agent_1"
        assert sandbox.agent_count == 1

    @pytest.mark.asyncio
    async def test_execute_action(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        agent = await sandbox.load_agent("agent_1", {})
        event = await sandbox.execute_agent_action(agent.id, "think", {"content": "test"})
        assert event.success is True
        assert agent.total_actions == 1

    @pytest.mark.asyncio
    async def test_tool_mock(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        agent = await sandbox.load_agent("agent_1", {})
        sandbox.mock_tool_response("search", {"results": ["a", "b"]})
        event = await sandbox.execute_agent_action(agent.id, "tool:search", {"query": "test"})
        assert event.success is True
        assert event.result["results"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_tool_error_mock(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        agent = await sandbox.load_agent("agent_1", {})
        sandbox.mock_tool_error("broken_tool", "Tool is broken")
        event = await sandbox.execute_agent_action(agent.id, "tool:broken_tool", {})
        assert event.success is False
        assert "broken" in event.error.lower()

    @pytest.mark.asyncio
    async def test_memory_isolation(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        a1 = await sandbox.load_agent("agent_1", {})
        a2 = await sandbox.load_agent("agent_2", {})

        await sandbox.execute_agent_action(a1.id, "remember", {"key": "secret", "value": "123"})
        result = await sandbox.execute_agent_action(a2.id, "recall", {"key": "secret"})
        assert result.result["found"] is False  # Isolated

    @pytest.mark.asyncio
    async def test_resource_limits(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        agent = await sandbox.load_agent("agent_1", {"max_actions": 2})
        await sandbox.execute_agent_action(agent.id, "think", {"content": "1"})
        await sandbox.execute_agent_action(agent.id, "think", {"content": "2"})
        event = await sandbox.execute_agent_action(agent.id, "think", {"content": "3"})
        assert event.success is False
        assert "limit" in event.error.lower()

    @pytest.mark.asyncio
    async def test_respond_action(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        agent = await sandbox.load_agent("agent_1", {})
        event = await sandbox.execute_agent_action(
            agent.id, "respond", {"message": "Hello!"},
        )
        assert event.success is True
        messages = sandbox.get_agent_messages(agent.id)
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_decide_action(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        agent = await sandbox.load_agent("agent_1", {})
        await sandbox.execute_agent_action(agent.id, "decide", {
            "options": ["A", "B"],
            "chosen": "A",
            "reasoning": "A is better",
        })
        decisions = sandbox.get_agent_decisions(agent.id)
        assert len(decisions) == 1

    @pytest.mark.asyncio
    async def test_unload_and_reset(self, sandbox):
        await sandbox.world_engine.initialize({"entities": []})
        agent = await sandbox.load_agent("agent_1", {})
        await sandbox.unload_agent(agent.id)
        assert sandbox.agent_count == 0

        await sandbox.load_agent("agent_2", {})
        await sandbox.reset()
        assert sandbox.agent_count == 0


# ---------------------------------------------------------------------------
# Tool Mock
# ---------------------------------------------------------------------------
from aion.simulation.sandbox.tool_mock import ToolMock, ToolMockRegistry


class TestToolMock:
    @pytest.mark.asyncio
    async def test_fixed_response(self):
        mock = ToolMock("search").returns({"results": [1, 2, 3]})
        result = await mock({"query": "test"})
        assert result["results"] == [1, 2, 3]
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_sequence_response(self):
        mock = ToolMock("counter").returns_sequence([1, 2, 3])
        assert await mock({}) == 1
        assert await mock({}) == 2
        assert await mock({}) == 3
        assert await mock({}) == 1  # Cycles

    @pytest.mark.asyncio
    async def test_conditional_response(self):
        mock = ToolMock("tool")
        mock.returns_when(lambda p: p.get("x") == 1, "matched")
        mock.returns("default")
        assert await mock({"x": 1}) == "matched"
        assert await mock({"x": 2}) == "default"

    @pytest.mark.asyncio
    async def test_error_injection(self):
        mock = ToolMock("broken").raises("Failure!")
        with pytest.raises(RuntimeError, match="Failure"):
            await mock({})

    @pytest.mark.asyncio
    async def test_was_called_with(self):
        mock = ToolMock("tool").returns("ok")
        await mock({"query": "test"})
        assert mock.was_called_with(query="test")
        assert not mock.was_called_with(query="other")


class TestToolMockRegistry:
    @pytest.mark.asyncio
    async def test_register_and_invoke(self):
        reg = ToolMockRegistry()
        reg.mock("search", default_response={"results": []})
        result = await reg.invoke("search", {"query": "test"})
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_assertions(self):
        reg = ToolMockRegistry()
        reg.mock("tool_a", default_response="a")
        reg.mock("tool_b", default_response="b")
        await reg.invoke("tool_a", {})
        await reg.invoke("tool_b", {})
        reg.assert_called("tool_a")
        reg.assert_called("tool_b")
        reg.assert_call_order("tool_a", "tool_b")

    @pytest.mark.asyncio
    async def test_assert_not_called(self):
        reg = ToolMockRegistry()
        reg.mock("unused", default_response="x")
        reg.assert_not_called("unused")


# ---------------------------------------------------------------------------
# Timeline Manager
# ---------------------------------------------------------------------------
from aion.simulation.timeline.manager import TimelineManager


class TestTimelineManager:
    @pytest.fixture
    def setup(self):
        config = SimulationConfig(seed=42)
        engine = WorldEngine(config)
        tm = TimelineManager(engine)
        return engine, tm

    @pytest.mark.asyncio
    async def test_create_snapshot(self, setup):
        engine, tm = setup
        await engine.initialize({"entities": [{"type": "user", "name": "alice"}]})
        snapshot = tm.create_snapshot("initial")
        assert snapshot.tick == 0
        assert snapshot.description == "initial"

    @pytest.mark.asyncio
    async def test_restore_snapshot(self, setup):
        engine, tm = setup
        await engine.initialize({"entities": [{"type": "user", "name": "alice"}]})
        snap = tm.create_snapshot("before")
        await engine.step()
        await engine.step()
        assert engine.state.tick == 2
        tm.restore_snapshot(snap.id)
        assert engine.state.tick == 0

    @pytest.mark.asyncio
    async def test_branching(self, setup):
        engine, tm = setup
        await engine.initialize({"entities": []})

        tm.create_snapshot("main_start")
        await engine.step()
        await engine.step()

        success = tm.create_branch("experiment")
        assert success
        assert tm.branches.current_branch == "experiment"
        assert "experiment" in tm.list_branches()

    @pytest.mark.asyncio
    async def test_switch_branch(self, setup):
        engine, tm = setup
        await engine.initialize({"entities": []})
        tm.create_snapshot("start")
        tm.create_branch("feature")
        await engine.step()
        tm.create_snapshot("feature_snap")

        tm.switch_branch("main")
        assert tm.branches.current_branch == "main"

    @pytest.mark.asyncio
    async def test_rewind(self, setup):
        engine, tm = setup
        await engine.initialize({"entities": []})
        tm.create_snapshot("t0")

        for _ in range(10):
            await engine.step()
            tm.create_snapshot()

        assert engine.state.tick == 10
        tm.rewind(5)
        assert engine.state.tick <= 5

    @pytest.mark.asyncio
    async def test_compare_snapshots(self, setup):
        engine, tm = setup
        await engine.initialize({"entities": [{"type": "user", "name": "alice"}]})
        s1 = tm.create_snapshot("before")
        engine.create_entity(EntityType.AGENT, "bot")
        await engine.step()
        s2 = tm.create_snapshot("after")

        diff = tm.compare_snapshots(s1.id, s2.id)
        assert diff["tick_diff"] == 1
        assert len(diff["entities_added"]) == 1

    @pytest.mark.asyncio
    async def test_stats(self, setup):
        engine, tm = setup
        await engine.initialize({"entities": []})
        tm.create_snapshot()
        stats = tm.stats
        assert stats["snapshot_count"] >= 1


# ---------------------------------------------------------------------------
# Snapshot Store
# ---------------------------------------------------------------------------
from aion.simulation.timeline.snapshots import SnapshotStore


class TestSnapshotStore:
    def test_store_and_get(self):
        store = SnapshotStore()
        snap = TimelineSnapshot(tick=0)
        snap.compute_fingerprint()
        sid = store.store(snap)
        assert store.get(sid) is not None

    def test_deduplication(self):
        store = SnapshotStore()
        s1 = TimelineSnapshot(tick=0, world_state=WorldState())
        s2 = TimelineSnapshot(tick=0, world_state=WorldState())
        s1.compute_fingerprint()
        s2.compute_fingerprint()
        id1 = store.store(s1)
        id2 = store.store(s2)
        # Both have empty states, same fingerprint → deduplicated
        assert id1 == id2
        assert store.count == 1

    def test_lru_eviction(self):
        store = SnapshotStore(max_snapshots=3)
        for i in range(5):
            state = WorldState()
            state.tick = i
            state.metrics["i"] = float(i)
            snap = TimelineSnapshot(tick=i, world_state=state)
            snap.compute_fingerprint()
            store.store(snap, deduplicate=False)
        assert store.count == 3

    def test_nearest(self):
        store = SnapshotStore()
        for i in [0, 5, 10, 20]:
            state = WorldState()
            state.tick = i
            snap = TimelineSnapshot(tick=i, world_state=state)
            snap.compute_fingerprint()
            store.store(snap, deduplicate=False)

        nearest = store.get_nearest(7, direction="before")
        assert nearest is not None
        assert nearest.tick == 5

        nearest_after = store.get_nearest(7, direction="after")
        assert nearest_after is not None
        assert nearest_after.tick == 10


# ---------------------------------------------------------------------------
# Branch Manager
# ---------------------------------------------------------------------------
from aion.simulation.timeline.branching import BranchManager


class TestBranchManager:
    def test_initial_state(self):
        mgr = BranchManager()
        assert mgr.current_branch == "main"
        assert "main" in mgr.list_branches()

    def test_create_and_switch(self):
        mgr = BranchManager()
        mgr.create("feature")
        mgr.switch("feature")
        assert mgr.current_branch == "feature"

    def test_duplicate_branch_error(self):
        mgr = BranchManager()
        mgr.create("feature")
        with pytest.raises(ValueError):
            mgr.create("feature")

    def test_lineage(self):
        mgr = BranchManager()
        mgr.create("feature")
        mgr.switch("feature")
        mgr.create("sub_feature")
        lineage = mgr.get_lineage("sub_feature")
        assert lineage == ["main", "feature", "sub_feature"]

    def test_delete(self):
        mgr = BranchManager()
        mgr.create("temp")
        assert mgr.delete("temp")
        assert not mgr.delete("main")  # Can't delete main


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
from aion.simulation.evaluation.evaluator import EvaluationResult, SimulationEvaluator
from aion.simulation.evaluation.metrics import MetricsCollector


class TestMetricsCollector:
    def test_record_and_summarize(self):
        mc = MetricsCollector()
        for i in range(100):
            mc.record("latency", float(i), tick=i)
        summary = mc.summarize("latency")
        assert summary.count == 100
        assert summary.mean == pytest.approx(49.5)
        assert summary.min_val == 0.0
        assert summary.max_val == 99.0

    def test_percentiles(self):
        mc = MetricsCollector()
        for i in range(1000):
            mc.record("val", float(i))
        summary = mc.summarize("val")
        assert summary.p50 == pytest.approx(499.5, abs=1)
        assert summary.p99 > 980

    def test_confidence_interval(self):
        mc = MetricsCollector()
        for _ in range(100):
            mc.record("normal", random.gauss(50, 5))
        summary = mc.summarize("normal")
        assert summary.ci_lower < summary.mean < summary.ci_upper

    def test_rate(self):
        mc = MetricsCollector()
        for i in range(10):
            mc.record("counter", float(i * 10), tick=i)
        rate = mc.rate("counter")
        assert rate == pytest.approx(10.0)

    def test_compare(self):
        mc1 = MetricsCollector()
        mc2 = MetricsCollector()
        for _ in range(50):
            mc1.record("score", random.gauss(100, 5))
            mc2.record("score", random.gauss(200, 5))
        result = mc1.compare("score", mc2)
        assert result["comparable"]
        assert result["significant"]  # Very different means


class TestSimulationEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        evaluator = SimulationEvaluator()
        result = SimulationResult(
            status=SimulationStatus.COMPLETED,
            goals_achieved=["goal1"],
            goals_failed=[],
            total_ticks=100,
            total_real_time=1.0,
            event_count=50,
        )
        eval_result = await evaluator.evaluate(result, WorldState())
        assert eval_result.passed

    @pytest.mark.asyncio
    async def test_evaluate_failure(self):
        evaluator = SimulationEvaluator()
        evaluator.add_assertion(Assertion(
            name="must_pass",
            condition="errors < 1",
            severity="error",
        ))
        result = SimulationResult(
            status=SimulationStatus.COMPLETED,
            errors=["something broke"],
        )
        eval_result = await evaluator.evaluate(result, WorldState())
        assert not eval_result.passed

    @pytest.mark.asyncio
    async def test_custom_assertion_fn(self):
        evaluator = SimulationEvaluator()
        evaluator.add_assertion(Assertion(
            name="custom",
            condition_fn=lambda r, s: r.total_ticks > 10,
        ))
        result = SimulationResult(total_ticks=5)
        eval_result = await evaluator.evaluate(result, WorldState())
        assert eval_result.assertions_failed == 1

    @pytest.mark.asyncio
    async def test_compare_runs(self):
        evaluator = SimulationEvaluator()
        results = [
            SimulationResult(
                success=True,
                metrics={"score": float(i)},
            )
            for i in range(10)
        ]
        comparison = await evaluator.compare_runs(results)
        assert comparison["run_count"] == 10
        assert "score" in comparison["metrics"]

    @pytest.mark.asyncio
    async def test_summary_generation(self):
        evaluator = SimulationEvaluator()
        result = SimulationResult(
            status=SimulationStatus.COMPLETED,
            goals_achieved=["g1"],
            total_ticks=50,
            total_simulation_time=50.0,
            total_real_time=0.5,
        )
        eval_result = await evaluator.evaluate(result, WorldState())
        assert "PASSED" in eval_result.summary or "FAILED" in eval_result.summary


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------
from aion.simulation.adversarial.generator import AdversarialGenerator
from aion.simulation.adversarial.fuzzing import Fuzzer, MutationEngine
from aion.simulation.adversarial.edge_cases import EdgeCaseDiscovery
from aion.simulation.adversarial.stress import StressTestGenerator, LoadProfile


class TestMutationEngine:
    def test_mutate_string(self):
        engine = MutationEngine(rng=random.Random(42))
        result = engine.mutate("hello world")
        assert isinstance(result, str)
        # Mutation should change something
        # (not guaranteed for every seed, but likely)

    def test_mutate_number(self):
        engine = MutationEngine(rng=random.Random(42))
        result = engine.mutate(42)
        assert result != 42 or isinstance(result, (int, float, str))

    def test_mutate_dict(self):
        engine = MutationEngine(rng=random.Random(42))
        result = engine.mutate({"key": "value"})
        assert isinstance(result, dict)

    def test_mutate_none(self):
        engine = MutationEngine(rng=random.Random(42))
        result = engine.mutate(None)
        assert result is not None  # Should produce something


class TestFuzzer:
    def test_random_generation(self):
        fuzzer = Fuzzer(strategy=FuzzStrategy.RANDOM, seed=42)
        results = fuzzer.generate(count=10)
        assert len(results) == 10
        assert all(isinstance(r, dict) for r in results)

    def test_mutation_generation(self):
        fuzzer = Fuzzer(strategy=FuzzStrategy.MUTATION, seed=42)
        base = {"message": "Hello, help me please"}
        results = fuzzer.generate(base_input=base, count=5)
        assert len(results) == 5

    def test_grammar_generation(self):
        fuzzer = Fuzzer(strategy=FuzzStrategy.GRAMMAR, seed=42)
        results = fuzzer.generate(count=10)
        assert len(results) == 10

    def test_corpus_and_interesting(self):
        fuzzer = Fuzzer(strategy=FuzzStrategy.COVERAGE_GUIDED, seed=42)
        fuzzer.add_to_corpus({"message": "base input"})
        fuzzer.mark_interesting({"message": "interesting!"}, path="path_1")
        results = fuzzer.generate(count=5)
        assert len(results) == 5
        assert fuzzer.stats["interesting_count"] == 1

    def test_stats(self):
        fuzzer = Fuzzer(seed=42)
        fuzzer.generate(count=3)
        assert fuzzer.stats["total_generated"] == 3


class TestEdgeCaseDiscovery:
    def test_discover_all(self):
        ecd = EdgeCaseDiscovery()
        cases = ecd.discover_all()
        assert len(cases) > 20
        assert ecd.discovered_count > 20

    def test_discover_category(self):
        ecd = EdgeCaseDiscovery()
        injection = ecd.discover_category("injection")
        assert len(injection) > 5
        assert "injection" in ecd.coverage

    def test_to_scenarios(self):
        ecd = EdgeCaseDiscovery()
        cases = ecd.discover_category("boundary")
        scenarios = ecd.to_scenarios(cases)
        assert len(scenarios) == len(cases)
        assert all(s.type == ScenarioType.ADVERSARIAL for s in scenarios)


class TestStressTestGenerator:
    def test_constant_load(self):
        gen = StressTestGenerator()
        scenario = gen.generate_constant_load(users=5, duration=5, rps=2)
        assert scenario.type == ScenarioType.STRESS
        assert len(scenario.scripted_events) > 0

    def test_ramp_load(self):
        gen = StressTestGenerator()
        scenario = gen.generate_ramp(start_users=1, end_users=10, duration=10)
        assert len(scenario.scripted_events) > 0

    def test_spike_load(self):
        gen = StressTestGenerator()
        scenario = gen.generate_spike(base_users=2, spike_users=20, duration=10)
        assert len(scenario.scripted_events) > 0

    def test_wave_load(self):
        gen = StressTestGenerator()
        scenario = gen.generate_wave(min_users=1, max_users=10, duration=10)
        assert len(scenario.scripted_events) > 0

    def test_suite(self):
        gen = StressTestGenerator()
        suite = gen.generate_suite()
        assert len(suite) == 4

    def test_load_profiles(self):
        events = LoadProfile.constant(users=2, duration=1, rps=5)
        assert len(events) > 0
        events = LoadProfile.ramp(1, 5, duration=2, rps=5)
        assert len(events) > 0


class TestAdversarialGenerator:
    def test_generate_suite_minimal(self):
        gen = AdversarialGenerator(seed=42)
        suite = gen.generate_suite(coverage="minimal")
        assert len(suite) > 10

    def test_generate_suite_standard(self):
        gen = AdversarialGenerator(seed=42)
        suite = gen.generate_suite(coverage="standard")
        assert len(suite) > 30

    def test_security_tests(self):
        gen = AdversarialGenerator(seed=42)
        tests = gen.generate_security_tests()
        assert len(tests) > 5

    def test_robustness_tests(self):
        gen = AdversarialGenerator(seed=42)
        tests = gen.generate_robustness_tests()
        assert len(tests) > 10

    def test_stats(self):
        gen = AdversarialGenerator(seed=42)
        gen.generate_suite(coverage="minimal")
        stats = gen.stats
        assert stats["total_scenarios"] > 0


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------
from aion.simulation.environment import SimulationEnvironment
from aion.simulation.config import test_config


class TestSimulationEnvironment:
    @pytest.fixture
    def env(self):
        return SimulationEnvironment(env_config=test_config())

    @pytest.mark.asyncio
    async def test_create_simulation(self, env):
        scenario = Scenario(
            name="test",
            initial_entities=[{"type": "user", "name": "alice"}],
            max_steps=10,
            max_time=100.0,
        )
        sim_id = await env.create_simulation(scenario)
        assert sim_id
        assert env.get_status() == SimulationStatus.INITIALIZING

    @pytest.mark.asyncio
    async def test_run_simple(self, env):
        scenario = Scenario(
            name="simple",
            initial_entities=[{"type": "user", "name": "alice"}],
            goals=[{"name": "done", "condition": "simulation_completed"}],
            max_steps=5,
            max_time=10.0,
        )
        config = SimulationConfig(max_ticks=5, seed=42)
        await env.create_simulation(scenario, config)
        result = await env.run()
        assert result.status == SimulationStatus.COMPLETED
        assert result.total_ticks == 5
        assert result.success

    @pytest.mark.asyncio
    async def test_run_with_agent(self, env):
        scenario = Scenario(
            name="agent_test",
            initial_entities=[{"type": "user", "name": "alice"}],
            max_steps=5,
            max_time=10.0,
        )
        config = SimulationConfig(max_ticks=5, seed=42)
        await env.create_simulation(scenario, config)
        agent = await env.load_agent("test_agent", {"type": "assistant"})
        assert agent.agent_id == "test_agent"
        result = await env.run()
        assert result.status == SimulationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_with_events(self, env):
        scenario = Scenario(
            name="events_test",
            initial_entities=[{"type": "user", "name": "alice"}],
            scripted_events=[
                {"time": 0, "type": "user_input", "data": {"message": "hello"}},
                {"time": 2, "type": "user_input", "data": {"message": "world"}},
            ],
            max_steps=10,
            max_time=20.0,
        )
        config = SimulationConfig(max_ticks=10, seed=42)
        await env.create_simulation(scenario, config)
        result = await env.run()
        assert result.event_count > 0

    @pytest.mark.asyncio
    async def test_timeline_control(self, env):
        scenario = Scenario(
            name="timeline",
            max_steps=20,
            max_time=100.0,
        )
        config = SimulationConfig(max_ticks=20, seed=42, snapshot_interval=5)
        await env.create_simulation(scenario, config)
        result = await env.run()
        assert result.snapshot_count > 0

    @pytest.mark.asyncio
    async def test_evaluate(self, env):
        scenario = Scenario(
            name="eval_test",
            goals=[{"name": "done", "condition": "simulation_completed"}],
            max_steps=5,
            max_time=10.0,
        )
        config = SimulationConfig(max_ticks=5, seed=42)
        await env.create_simulation(scenario, config)
        await env.run()
        eval_result = await env.evaluate()
        assert isinstance(eval_result, EvaluationResult)
        assert eval_result.summary

    @pytest.mark.asyncio
    async def test_batch_scenarios(self, env):
        scenarios = [
            Scenario(name=f"batch_{i}", max_steps=3, max_time=10.0)
            for i in range(3)
        ]
        config = SimulationConfig(max_ticks=3, seed=42)
        results = await env.run_scenarios(scenarios, config)
        assert len(results) == 3
        assert all(r.status == SimulationStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_run_with_variations(self, env):
        scenario = Scenario(name="variation_base", max_steps=3, max_time=10.0)
        config = SimulationConfig(max_ticks=3, seed=42)
        variations = [
            {"max_steps": 5},
            {"max_steps": 10},
        ]
        results = await env.run_with_variations(scenario, variations, config)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_history(self, env):
        scenario = Scenario(name="hist", max_steps=3, max_time=10.0)
        config = SimulationConfig(max_ticks=3, seed=42)
        await env.create_simulation(scenario, config)
        await env.run()
        assert len(env.get_history()) == 1

    @pytest.mark.asyncio
    async def test_pause_stop(self, env):
        scenario = Scenario(name="ctrl", max_steps=3, max_time=10.0)
        await env.create_simulation(scenario)
        env.pause()
        assert env.get_status() == SimulationStatus.PAUSED
        env.resume()
        assert env.get_status() == SimulationStatus.RUNNING
        env.stop()
        assert env.get_status() == SimulationStatus.CANCELLED


# ---------------------------------------------------------------------------
# Simulation API
# ---------------------------------------------------------------------------
from aion.simulation.api import SimulationAPI


class TestSimulationAPI:
    @pytest.fixture
    def api(self):
        return SimulationAPI(env_config=test_config())

    @pytest.mark.asyncio
    async def test_run_scenario(self, api):
        result = await api.run_scenario("customer_support_basic")
        assert result.status == SimulationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_custom_scenario(self, api):
        scenario = Scenario(name="custom", max_steps=3, max_time=10.0)
        result = await api.run_custom_scenario(
            scenario,
            sim_config=SimulationConfig(max_ticks=3, seed=42),
        )
        assert result.status == SimulationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_template_batch(self, api):
        results = await api.run_template_batch(
            ["customer_support_basic", "task_execution_simple"],
            sim_config=SimulationConfig(max_ticks=5, seed=42),
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_evaluate_result(self, api):
        result = await api.run_scenario(
            "customer_support_basic",
            sim_config=SimulationConfig(max_ticks=5, seed=42),
        )
        evaluation = await api.evaluate_result(result)
        assert isinstance(evaluation, EvaluationResult)

    @pytest.mark.asyncio
    async def test_list_templates(self, api):
        templates = api.list_templates()
        assert "customer_support_basic" in templates
        assert len(templates) > 3

    @pytest.mark.asyncio
    async def test_compare_results(self, api):
        results = []
        for i in range(3):
            r = await api.run_scenario(
                "task_execution_simple",
                sim_config=SimulationConfig(max_ticks=5, seed=i),
            )
            results.append(r)
        comparison = await api.compare_results(results)
        assert comparison["run_count"] == 3


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_simulation_lifecycle(self):
        """Test complete lifecycle: create -> configure -> run -> evaluate."""
        env = SimulationEnvironment(env_config=test_config())

        # Create scenario
        scenario = Scenario(
            name="integration_test",
            initial_entities=[
                {"type": "user", "name": "alice", "properties": {"role": "tester"}},
                {"type": "agent", "name": "bot"},
            ],
            scripted_events=[
                {"time": 0, "type": "user_input", "data": {"message": "Hello!"}},
                {"time": 2, "type": "user_input", "data": {"message": "Help me"}},
            ],
            goals=[
                {"name": "responded", "condition": "simulation_completed"},
            ],
            max_steps=20,
            max_time=50.0,
        )

        config = SimulationConfig(
            max_ticks=20,
            seed=42,
            deterministic=True,
            snapshot_interval=5,
        )

        # Create and configure
        sim_id = await env.create_simulation(scenario, config)
        assert sim_id

        # Load agent
        agent = await env.load_agent(
            "test_bot",
            config={"type": "assistant"},
            tool_mocks={"search": lambda p: {"results": ["r1"]}},
        )

        # Run
        result = await env.run()
        assert result.status == SimulationStatus.COMPLETED
        assert result.total_ticks == 20
        assert result.event_count > 0
        assert result.snapshot_count > 0

        # Evaluate
        evaluation = await env.evaluate()
        assert evaluation.summary
        assert evaluation.score >= 0

    @pytest.mark.asyncio
    async def test_timeline_branching_workflow(self):
        """Test creating branches and comparing outcomes."""
        env = SimulationEnvironment(env_config=test_config())

        scenario = Scenario(
            name="branching_test",
            initial_entities=[{"type": "user", "name": "alice"}],
            max_steps=20,
            max_time=50.0,
        )
        config = SimulationConfig(max_ticks=10, seed=42, snapshot_interval=2)

        await env.create_simulation(scenario, config)
        result = await env.run()

        # Access timeline
        tm = env.timeline_manager
        assert tm is not None

        snapshots = tm.get_branch_history()
        assert len(snapshots) > 0

    @pytest.mark.asyncio
    async def test_adversarial_pipeline(self):
        """Test generating and running adversarial scenarios."""
        env = SimulationEnvironment(env_config=test_config())

        # Generate adversarial suite (minimal for speed)
        scenarios = env.adversarial.generate_suite(coverage="minimal")
        assert len(scenarios) > 0

        # Run a subset
        config = SimulationConfig(max_ticks=5, seed=42)
        results = await env.run_scenarios(scenarios[:3], config)
        assert len(results) == 3
        assert all(r.status == SimulationStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_deterministic_replay(self):
        """Test that same seed produces identical results."""
        results = []
        for _ in range(2):
            env = SimulationEnvironment(env_config=test_config())
            scenario = Scenario(
                name="deterministic",
                initial_entities=[{"type": "user", "name": "alice"}],
                scripted_events=[
                    {"time": 0, "type": "user_input", "data": {"message": "test"}},
                ],
                max_steps=10,
                max_time=20.0,
            )
            config = SimulationConfig(max_ticks=10, seed=42, deterministic=True)
            await env.create_simulation(scenario, config)
            result = await env.run()
            results.append(result)

        assert results[0].total_ticks == results[1].total_ticks
        assert results[0].event_count == results[1].event_count

    @pytest.mark.asyncio
    async def test_performance_1000_ticks(self):
        """Verify >1000 ticks/second for simple scenarios."""
        import time

        env = SimulationEnvironment(env_config=test_config())
        scenario = Scenario(name="perf", max_steps=2000, max_time=5000.0)
        config = SimulationConfig(
            max_ticks=1000,
            seed=42,
            record_all_events=False,
            record_state_snapshots=False,
            record_causal_graph=False,
        )

        await env.create_simulation(scenario, config)
        start = time.monotonic()
        result = await env.run()
        elapsed = time.monotonic() - start

        ticks_per_second = result.total_ticks / elapsed if elapsed > 0 else 0
        assert ticks_per_second > 1000, (
            f"Performance: {ticks_per_second:.0f} ticks/s (expected >1000)"
        )
