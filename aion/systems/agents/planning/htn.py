"""
Hierarchical Task Network (HTN) Planner

Implements HTN planning for decomposing complex tasks into
executable action sequences.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import structlog

logger = structlog.get_logger()


@dataclass
class Operator:
    """A primitive action that can be executed."""

    name: str
    preconditions: dict[str, Any] = field(default_factory=dict)
    effects: dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    duration: float = 1.0

    def is_applicable(self, state: dict[str, Any]) -> bool:
        """Check if operator is applicable in state."""
        for key, value in self.preconditions.items():
            if state.get(key) != value:
                return False
        return True

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply operator to state."""
        new_state = state.copy()
        new_state.update(self.effects)
        return new_state


@dataclass
class Task:
    """A task that needs to be accomplished."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    primitive: bool = False  # If True, maps directly to an operator
    priority: float = 0.5

    def __hash__(self):
        return hash((self.name, tuple(sorted(self.parameters.items()))))


@dataclass
class Method:
    """A method for decomposing a compound task."""

    name: str
    task_name: str  # Which task this decomposes
    preconditions: dict[str, Any] = field(default_factory=dict)
    subtasks: list[Task] = field(default_factory=list)
    ordered: bool = True  # If True, subtasks must be done in order

    def is_applicable(self, state: dict[str, Any], task: Task) -> bool:
        """Check if method is applicable."""
        if task.name != self.task_name:
            return False
        for key, value in self.preconditions.items():
            if state.get(key) != value:
                return False
        return True


@dataclass
class Plan:
    """A plan produced by HTN planning."""

    id: str
    tasks: list[Task]
    operators: list[Operator] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration: float = 0.0
    success: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "tasks": [{"name": t.name, "params": t.parameters} for t in self.tasks],
            "operators": [o.name for o in self.operators],
            "total_cost": self.total_cost,
            "total_duration": self.total_duration,
            "success": self.success,
        }


class HTNPlanner:
    """
    Hierarchical Task Network planner.

    Features:
    - Task decomposition
    - Method selection
    - Plan optimization
    - Backtracking search
    """

    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self._operators: dict[str, Operator] = {}
        self._methods: list[Method] = []
        self._plan_counter = 0

    def add_operator(self, operator: Operator) -> None:
        """Add a primitive operator."""
        self._operators[operator.name] = operator

    def add_method(self, method: Method) -> None:
        """Add a decomposition method."""
        self._methods.append(method)

    def plan(
        self,
        initial_state: dict[str, Any],
        goal_tasks: list[Task],
    ) -> Plan:
        """Generate a plan to accomplish goal tasks."""
        self._plan_counter += 1
        plan = Plan(
            id=f"plan-{self._plan_counter}",
            tasks=goal_tasks.copy(),
        )

        # Decompose tasks
        operators = self._decompose(initial_state, goal_tasks, 0)

        if operators is not None:
            plan.operators = operators
            plan.success = True
            plan.total_cost = sum(op.cost for op in operators)
            plan.total_duration = sum(op.duration for op in operators)

        logger.info(
            "htn_plan_generated",
            success=plan.success,
            operators=len(plan.operators),
        )

        return plan

    def _decompose(
        self,
        state: dict[str, Any],
        tasks: list[Task],
        depth: int,
    ) -> Optional[list[Operator]]:
        """Recursively decompose tasks into operators."""
        if depth > self.max_depth:
            return None

        if not tasks:
            return []

        task = tasks[0]
        remaining = tasks[1:]

        # Check if primitive task
        if task.primitive or task.name in self._operators:
            operator = self._operators.get(task.name)
            if operator and operator.is_applicable(state):
                new_state = operator.apply(state)
                rest = self._decompose(new_state, remaining, depth + 1)
                if rest is not None:
                    return [operator] + rest

        # Try decomposition methods
        for method in self._methods:
            if method.is_applicable(state, task):
                if method.ordered:
                    new_tasks = method.subtasks + remaining
                else:
                    new_tasks = remaining + method.subtasks

                result = self._decompose(state, new_tasks, depth + 1)
                if result is not None:
                    return result

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get planner statistics."""
        return {
            "operators": len(self._operators),
            "methods": len(self._methods),
            "plans_generated": self._plan_counter,
        }
