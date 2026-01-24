"""
Goal-Oriented Action Planning (GOAP)

Implements GOAP for dynamic goal-driven planning with
A* search over action space.
"""

import heapq
from dataclasses import dataclass, field
from typing import Any, Optional
import structlog

logger = structlog.get_logger()


@dataclass
class WorldState:
    """Represents the world state as a set of facts."""

    facts: dict[str, Any] = field(default_factory=dict)

    def satisfies(self, conditions: dict[str, Any]) -> bool:
        """Check if state satisfies conditions."""
        for key, value in conditions.items():
            if self.facts.get(key) != value:
                return False
        return True

    def apply_effects(self, effects: dict[str, Any]) -> "WorldState":
        """Create new state with effects applied."""
        new_facts = self.facts.copy()
        new_facts.update(effects)
        return WorldState(facts=new_facts)

    def distance_to(self, goal: dict[str, Any]) -> int:
        """Heuristic distance to goal state."""
        unsatisfied = 0
        for key, value in goal.items():
            if self.facts.get(key) != value:
                unsatisfied += 1
        return unsatisfied

    def __hash__(self):
        return hash(tuple(sorted(self.facts.items())))

    def __eq__(self, other):
        if isinstance(other, WorldState):
            return self.facts == other.facts
        return False


@dataclass
class Action:
    """An action in GOAP."""

    name: str
    preconditions: dict[str, Any] = field(default_factory=dict)
    effects: dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0

    def is_valid(self, state: WorldState) -> bool:
        """Check if action is valid in state."""
        return state.satisfies(self.preconditions)

    def execute(self, state: WorldState) -> WorldState:
        """Execute action and return new state."""
        return state.apply_effects(self.effects)


@dataclass
class Goal:
    """A goal to achieve."""

    name: str
    conditions: dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0

    def is_satisfied(self, state: WorldState) -> bool:
        """Check if goal is satisfied."""
        return state.satisfies(self.conditions)


@dataclass
class GOAPPlan:
    """A plan produced by GOAP."""

    goal: Goal
    actions: list[Action] = field(default_factory=list)
    total_cost: float = 0.0
    valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal": self.goal.name,
            "actions": [a.name for a in self.actions],
            "total_cost": self.total_cost,
            "valid": self.valid,
        }


class GOAPPlanner:
    """
    Goal-Oriented Action Planner.

    Features:
    - A* search for optimal plans
    - Dynamic action availability
    - Goal prioritization
    - Plan caching
    """

    def __init__(self, max_iterations: int = 1000):
        self.max_iterations = max_iterations
        self._actions: list[Action] = []
        self._plan_cache: dict[tuple, GOAPPlan] = {}

    def add_action(self, action: Action) -> None:
        """Add an action to the planner."""
        self._actions.append(action)

    def plan(
        self,
        initial_state: WorldState,
        goal: Goal,
        use_cache: bool = True,
    ) -> GOAPPlan:
        """
        Find a plan to achieve the goal.

        Uses A* search backwards from goal to current state.
        """
        # Check cache
        cache_key = (hash(initial_state), goal.name)
        if use_cache and cache_key in self._plan_cache:
            return self._plan_cache[cache_key]

        # Check if goal already satisfied
        if goal.is_satisfied(initial_state):
            plan = GOAPPlan(goal=goal, valid=True)
            self._plan_cache[cache_key] = plan
            return plan

        # A* search
        plan = self._astar_search(initial_state, goal)

        if use_cache:
            self._plan_cache[cache_key] = plan

        logger.info(
            "goap_plan_generated",
            goal=goal.name,
            valid=plan.valid,
            actions=len(plan.actions),
        )

        return plan

    def _astar_search(
        self,
        initial_state: WorldState,
        goal: Goal,
    ) -> GOAPPlan:
        """A* search for plan."""
        # Priority queue: (f_score, counter, state, actions)
        counter = 0
        start_h = initial_state.distance_to(goal.conditions)
        open_set = [(start_h, counter, initial_state, [])]
        closed_set = set()

        for _ in range(self.max_iterations):
            if not open_set:
                break

            f_score, _, current_state, actions = heapq.heappop(open_set)

            if goal.is_satisfied(current_state):
                return GOAPPlan(
                    goal=goal,
                    actions=actions,
                    total_cost=sum(a.cost for a in actions),
                    valid=True,
                )

            state_hash = hash(current_state)
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)

            # Try each action
            for action in self._actions:
                if action.is_valid(current_state):
                    new_state = action.execute(current_state)
                    new_state_hash = hash(new_state)

                    if new_state_hash in closed_set:
                        continue

                    new_actions = actions + [action]
                    g_score = sum(a.cost for a in new_actions)
                    h_score = new_state.distance_to(goal.conditions)
                    f_score = g_score + h_score

                    counter += 1
                    heapq.heappush(
                        open_set,
                        (f_score, counter, new_state, new_actions),
                    )

        # No plan found
        return GOAPPlan(goal=goal, valid=False)

    def find_best_goal(
        self,
        state: WorldState,
        goals: list[Goal],
    ) -> Optional[Goal]:
        """Find the best achievable goal."""
        best_goal = None
        best_score = float("-inf")

        for goal in goals:
            if goal.is_satisfied(state):
                continue

            plan = self.plan(state, goal)
            if plan.valid:
                # Score based on priority and cost
                score = goal.priority / (plan.total_cost + 1)
                if score > best_score:
                    best_score = score
                    best_goal = goal

        return best_goal

    def get_stats(self) -> dict[str, Any]:
        """Get planner statistics."""
        return {
            "actions": len(self._actions),
            "cached_plans": len(self._plan_cache),
        }
