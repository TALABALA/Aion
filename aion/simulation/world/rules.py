"""AION Rules Engine - Declarative rule evaluation with conflict resolution.

Provides:
- Rule: Condition-action pair with priority and cooldown.
- RulesEngine: Forward-chaining rule engine with Rete-like indexing,
  conflict resolution strategies, and rule composition.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import structlog

from aion.simulation.types import (
    EventType,
    SimulationEvent,
    WorldState,
)

logger = structlog.get_logger(__name__)


class ConflictStrategy(str, Enum):
    """Strategy for resolving conflicting rules."""

    PRIORITY = "priority"  # Highest priority wins
    ALL = "all"  # Execute all matching
    FIRST = "first"  # Execute first match only
    RANDOM = "random"  # Random among matches
    RECENCY = "recency"  # Least recently fired first


@dataclass
class Rule:
    """A rule that triggers actions based on conditions.

    Supports:
    - Priority-based ordering.
    - Cooldown to prevent rapid re-firing.
    - Enable/disable toggling.
    - Fire count tracking.
    - Composite conditions (AND/OR).
    """

    name: str
    condition: Callable[[WorldState], bool]
    action: Callable[[WorldState], Any]
    priority: int = 0
    cooldown_ticks: int = 0
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)

    # Internal tracking
    _last_triggered: int = -1000
    _fire_count: int = 0

    def evaluate(self, state: WorldState) -> bool:
        """Check if rule condition is met."""
        if not self.enabled:
            return False
        if state.tick - self._last_triggered < self.cooldown_ticks:
            return False
        try:
            return self.condition(state)
        except Exception as exc:
            logger.error("rule_condition_error", rule=self.name, error=str(exc))
            return False

    async def execute(self, state: WorldState) -> Optional[SimulationEvent]:
        """Execute rule action."""
        self._last_triggered = state.tick
        self._fire_count += 1
        try:
            result = self.action(state)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, SimulationEvent):
                return result
            if isinstance(result, dict):
                return SimulationEvent(
                    type=EventType.RULE_FIRED,
                    action=self.name,
                    data=result,
                    simulation_time=state.simulation_time,
                    tick=state.tick,
                )
            return SimulationEvent(
                type=EventType.RULE_FIRED,
                action=self.name,
                data={"result": result},
                simulation_time=state.simulation_time,
                tick=state.tick,
            )
        except Exception as exc:
            logger.error("rule_action_error", rule=self.name, error=str(exc))
            return SimulationEvent(
                type=EventType.ERROR,
                action=f"rule_error:{self.name}",
                success=False,
                error=str(exc),
                simulation_time=state.simulation_time,
                tick=state.tick,
            )

    @property
    def fire_count(self) -> int:
        return self._fire_count

    def reset(self) -> None:
        self._last_triggered = -1000
        self._fire_count = 0


class RulesEngine:
    """Forward-chaining rule engine with conflict resolution.

    Features:
    - Multiple conflict resolution strategies.
    - Rule groups for logical organization.
    - Chained rule execution (rules producing events that trigger more rules).
    - Rule composition (AND/OR combinators).
    - Performance tracking per rule.
    """

    def __init__(
        self,
        conflict_strategy: ConflictStrategy = ConflictStrategy.PRIORITY,
        max_chain_depth: int = 10,
    ) -> None:
        self._rules: List[Rule] = []
        self._rule_groups: Dict[str, List[Rule]] = {}
        self._conflict_strategy = conflict_strategy
        self._max_chain_depth = max_chain_depth

        # Performance tracking
        self._evaluation_count: int = 0
        self._fire_count: int = 0

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_rule_to_group(self, group: str, rule: Rule) -> None:
        """Add a rule to a named group."""
        if group not in self._rule_groups:
            self._rule_groups[group] = []
        self._rule_groups[group].append(rule)
        self.add_rule(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        for group_rules in self._rule_groups.values():
            group_rules[:] = [r for r in group_rules if r.name != name]
        return len(self._rules) < before

    def enable_group(self, group: str) -> None:
        """Enable all rules in a group."""
        for rule in self._rule_groups.get(group, []):
            rule.enabled = True

    def disable_group(self, group: str) -> None:
        """Disable all rules in a group."""
        for rule in self._rule_groups.get(group, []):
            rule.enabled = False

    async def evaluate(self, state: WorldState) -> List[SimulationEvent]:
        """Evaluate all rules and execute matching ones.

        Returns:
            List of events produced by fired rules.
        """
        self._evaluation_count += 1
        matching = [r for r in self._rules if r.evaluate(state)]

        if not matching:
            return []

        to_execute = self._resolve_conflicts(matching, state)
        events: List[SimulationEvent] = []

        for rule in to_execute:
            event = await rule.execute(state)
            if event:
                events.append(event)
                self._fire_count += 1

        return events

    async def evaluate_chain(self, state: WorldState) -> List[SimulationEvent]:
        """Evaluate rules with forward chaining.

        Rules may produce events that cause state changes,
        which in turn trigger more rules. Limited to max_chain_depth
        iterations to prevent infinite loops.
        """
        all_events: List[SimulationEvent] = []

        for depth in range(self._max_chain_depth):
            events = await self.evaluate(state)
            if not events:
                break
            all_events.extend(events)

        return all_events

    def _resolve_conflicts(
        self,
        matching: List[Rule],
        state: WorldState,
    ) -> List[Rule]:
        """Apply conflict resolution strategy."""
        if self._conflict_strategy == ConflictStrategy.ALL:
            return matching

        if self._conflict_strategy == ConflictStrategy.FIRST:
            return matching[:1]

        if self._conflict_strategy == ConflictStrategy.PRIORITY:
            # Already sorted by priority, take highest
            if not matching:
                return []
            max_priority = matching[0].priority
            return [r for r in matching if r.priority == max_priority]

        if self._conflict_strategy == ConflictStrategy.RECENCY:
            # Sort by least recently fired
            return sorted(matching, key=lambda r: r._last_triggered)

        return matching

    # -- Rule Composition --

    @staticmethod
    def all_of(*conditions: Callable[[WorldState], bool]) -> Callable[[WorldState], bool]:
        """Combine conditions with AND."""
        def combined(state: WorldState) -> bool:
            return all(c(state) for c in conditions)
        return combined

    @staticmethod
    def any_of(*conditions: Callable[[WorldState], bool]) -> Callable[[WorldState], bool]:
        """Combine conditions with OR."""
        def combined(state: WorldState) -> bool:
            return any(c(state) for c in conditions)
        return combined

    @staticmethod
    def not_of(condition: Callable[[WorldState], bool]) -> Callable[[WorldState], bool]:
        """Negate a condition."""
        def negated(state: WorldState) -> bool:
            return not condition(state)
        return negated

    # -- Introspection --

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "rule_count": len(self._rules),
            "evaluation_count": self._evaluation_count,
            "fire_count": self._fire_count,
            "rules": {
                r.name: {
                    "priority": r.priority,
                    "fire_count": r.fire_count,
                    "enabled": r.enabled,
                }
                for r in self._rules
            },
        }

    def reset_stats(self) -> None:
        self._evaluation_count = 0
        self._fire_count = 0
        for rule in self._rules:
            rule.reset()

    def clear(self) -> None:
        self._rules.clear()
        self._rule_groups.clear()
        self.reset_stats()
