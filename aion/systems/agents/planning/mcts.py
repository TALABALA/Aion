"""
Monte Carlo Tree Search (MCTS) Planner

Implements MCTS for planning under uncertainty with
simulation-based action selection.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Callable
import structlog

logger = structlog.get_logger()


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""

    exploration_constant: float = 1.414
    max_iterations: int = 1000
    max_depth: int = 20
    discount_factor: float = 0.99
    simulation_depth: int = 10


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""

    id: str
    state: dict[str, Any]
    action: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    depth: int = 0

    @property
    def value(self) -> float:
        """Average value of this node."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, parent_visits: int, c: float) -> float:
        """Calculate UCB1 score."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.value
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner.

    Features:
    - UCB1 selection
    - Random rollout simulation
    - Backpropagation
    - Best action selection
    """

    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        get_actions_fn: Optional[Callable[[dict], list[str]]] = None,
        transition_fn: Optional[Callable[[dict, str], dict]] = None,
        reward_fn: Optional[Callable[[dict], float]] = None,
        is_terminal_fn: Optional[Callable[[dict], bool]] = None,
    ):
        self.config = config or MCTSConfig()
        self.get_actions = get_actions_fn or (lambda s: ["action1", "action2"])
        self.transition = transition_fn or (lambda s, a: s)
        self.get_reward = reward_fn or (lambda s: 0.0)
        self.is_terminal = is_terminal_fn or (lambda s: False)

        self._nodes: dict[str, MCTSNode] = {}
        self._node_counter = 0

    def plan(self, initial_state: dict[str, Any]) -> tuple[str, float]:
        """
        Plan the best action from initial state.

        Returns:
            Tuple of (best action, expected value)
        """
        # Create root node
        root = self._create_node(initial_state, None, None)
        root_id = root.id

        # Run MCTS iterations
        for _ in range(self.config.max_iterations):
            # Selection
            node_id = self._select(root_id)
            node = self._nodes[node_id]

            # Expansion
            if node.visits > 0 and not self.is_terminal(node.state):
                node_id = self._expand(node_id)
                node = self._nodes[node_id]

            # Simulation
            value = self._simulate(node.state, node.depth)

            # Backpropagation
            self._backpropagate(node_id, value)

        # Select best action
        root = self._nodes[root_id]
        best_child_id = None
        best_visits = -1

        for child_id in root.children_ids:
            child = self._nodes[child_id]
            if child.visits > best_visits:
                best_visits = child.visits
                best_child_id = child_id

        if best_child_id:
            best_child = self._nodes[best_child_id]
            return best_child.action or "", best_child.value

        return "", 0.0

    def _create_node(
        self,
        state: dict[str, Any],
        action: Optional[str],
        parent_id: Optional[str],
    ) -> MCTSNode:
        """Create a new node."""
        self._node_counter += 1
        depth = 0
        if parent_id and parent_id in self._nodes:
            depth = self._nodes[parent_id].depth + 1

        node = MCTSNode(
            id=f"mcts-{self._node_counter}",
            state=state,
            action=action,
            parent_id=parent_id,
            depth=depth,
        )
        self._nodes[node.id] = node

        if parent_id and parent_id in self._nodes:
            self._nodes[parent_id].children_ids.append(node.id)

        return node

    def _select(self, node_id: str) -> str:
        """Select a node to expand using UCB1."""
        node = self._nodes[node_id]

        while node.children_ids:
            # Check if fully expanded
            if node.visits == 0:
                return node.id

            actions = self.get_actions(node.state)
            if len(node.children_ids) < len(actions):
                return node.id

            # Select child with best UCB score
            best_child_id = None
            best_score = float("-inf")

            for child_id in node.children_ids:
                child = self._nodes[child_id]
                score = child.ucb_score(node.visits, self.config.exploration_constant)
                if score > best_score:
                    best_score = score
                    best_child_id = child_id

            if best_child_id:
                node = self._nodes[best_child_id]
            else:
                break

        return node.id

    def _expand(self, node_id: str) -> str:
        """Expand a node by adding a child."""
        node = self._nodes[node_id]

        # Get untried actions
        actions = self.get_actions(node.state)
        tried_actions = {
            self._nodes[cid].action
            for cid in node.children_ids
        }
        untried = [a for a in actions if a not in tried_actions]

        if not untried:
            return node_id

        # Choose random untried action
        action = random.choice(untried)
        new_state = self.transition(node.state, action)

        child = self._create_node(new_state, action, node_id)
        return child.id

    def _simulate(self, state: dict[str, Any], depth: int) -> float:
        """Simulate a random rollout from state."""
        total_reward = 0.0
        current_state = state.copy()
        discount = 1.0

        for _ in range(self.config.simulation_depth):
            if self.is_terminal(current_state):
                break

            actions = self.get_actions(current_state)
            if not actions:
                break

            action = random.choice(actions)
            current_state = self.transition(current_state, action)

            reward = self.get_reward(current_state)
            total_reward += discount * reward
            discount *= self.config.discount_factor

        return total_reward

    def _backpropagate(self, node_id: str, value: float) -> None:
        """Backpropagate value up the tree."""
        current_id = node_id

        while current_id:
            node = self._nodes[current_id]
            node.visits += 1
            node.total_value += value

            value *= self.config.discount_factor
            current_id = node.parent_id

    def get_stats(self) -> dict[str, Any]:
        """Get planner statistics."""
        return {
            "nodes_created": len(self._nodes),
            "config": {
                "exploration_constant": self.config.exploration_constant,
                "max_iterations": self.config.max_iterations,
            },
        }
