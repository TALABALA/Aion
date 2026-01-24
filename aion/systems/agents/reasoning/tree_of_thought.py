"""
Tree-of-Thought Reasoning

Implementation of Tree-of-Thought (ToT) prompting for complex
problem-solving through structured exploration of reasoning paths.

Based on: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
"""

import asyncio
import heapq
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

import structlog

logger = structlog.get_logger()


class SearchStrategy(Enum):
    """Search strategies for tree exploration."""

    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search
    BEAM = "beam"  # Beam search
    MCTS = "mcts"  # Monte Carlo Tree Search
    BEST_FIRST = "best_first"  # Best-first search


class NodeState(Enum):
    """States of thought nodes."""

    PENDING = "pending"
    EXPLORING = "exploring"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    TERMINAL = "terminal"


@dataclass
class ThoughtNode:
    """A node in the thought tree."""

    id: str
    thought: str  # The thought/reasoning step
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    state: NodeState = NodeState.PENDING

    # Evaluation
    value: float = 0.0  # Estimated value of this path
    confidence: float = 0.5
    visits: int = 0

    # MCTS specific
    total_reward: float = 0.0
    ucb_score: float = 0.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children_ids) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "thought": self.thought,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "depth": self.depth,
            "state": self.state.value,
            "value": self.value,
            "confidence": self.confidence,
            "visits": self.visits,
            "total_reward": self.total_reward,
            "ucb_score": self.ucb_score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ThoughtTree:
    """The complete thought tree."""

    root_id: str
    nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    best_path: list[str] = field(default_factory=list)
    best_value: float = 0.0
    total_nodes: int = 0
    max_depth_reached: int = 0

    def add_node(self, node: ThoughtNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        self.total_nodes += 1
        self.max_depth_reached = max(self.max_depth_reached, node.depth)

    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_path(self, node_id: str) -> list[ThoughtNode]:
        """Get the path from root to a node."""
        path = []
        current = self.nodes.get(node_id)

        while current:
            path.append(current)
            if current.parent_id:
                current = self.nodes.get(current.parent_id)
            else:
                break

        return list(reversed(path))

    def get_path_thoughts(self, node_id: str) -> list[str]:
        """Get the thoughts along the path to a node."""
        return [node.thought for node in self.get_path(node_id)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root_id": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "best_path": self.best_path,
            "best_value": self.best_value,
            "total_nodes": self.total_nodes,
            "max_depth_reached": self.max_depth_reached,
        }


@dataclass
class ToTConfig:
    """Configuration for Tree-of-Thought."""

    # Tree structure
    max_depth: int = 5
    branching_factor: int = 3  # Number of thoughts to generate per node
    max_nodes: int = 100

    # Search
    search_strategy: SearchStrategy = SearchStrategy.BEAM
    beam_width: int = 3

    # MCTS specific
    exploration_constant: float = 1.414  # UCB exploration constant
    simulation_depth: int = 3
    num_simulations: int = 10

    # Evaluation
    value_threshold: float = 0.3  # Prune nodes below this value
    early_stop_value: float = 0.95  # Stop if we find a path this good

    # Generation
    temperature: float = 0.7
    thought_prompt_template: str = """Given the problem and current reasoning path, generate the next step of reasoning.

Problem: {problem}

Current reasoning path:
{path}

Generate the next logical thought step:"""

    evaluation_prompt_template: str = """Evaluate how promising this reasoning path is for solving the problem.

Problem: {problem}

Reasoning path:
{path}

Rate this path from 0.0 (completely wrong/unhelpful) to 1.0 (correct and complete solution).
Provide only the numerical score:"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "search_strategy": self.search_strategy.value,
            "beam_width": self.beam_width,
            "exploration_constant": self.exploration_constant,
            "value_threshold": self.value_threshold,
            "early_stop_value": self.early_stop_value,
        }


# Type for LLM generation function
GenerateFn = Callable[[str], Awaitable[str]]
EvaluateFn = Callable[[str, list[str]], Awaitable[float]]


class TreeOfThought:
    """
    Tree-of-Thought reasoning system.

    Features:
    - Multiple search strategies (BFS, DFS, Beam, MCTS, Best-First)
    - Dynamic thought generation
    - Path evaluation and pruning
    - Backtracking support
    - Parallel exploration
    """

    def __init__(
        self,
        config: Optional[ToTConfig] = None,
        generate_fn: Optional[GenerateFn] = None,
        evaluate_fn: Optional[EvaluateFn] = None,
    ):
        self.config = config or ToTConfig()
        self.generate_fn = generate_fn
        self.evaluate_fn = evaluate_fn

        self._node_counter = 0
        self._current_tree: Optional[ThoughtTree] = None

    async def solve(
        self,
        problem: str,
        initial_thoughts: Optional[list[str]] = None,
    ) -> tuple[list[str], float, ThoughtTree]:
        """
        Solve a problem using Tree-of-Thought reasoning.

        Args:
            problem: The problem to solve
            initial_thoughts: Optional initial reasoning steps

        Returns:
            Tuple of (best reasoning path, value, complete tree)
        """
        # Initialize tree
        root_id = self._generate_node_id()
        root = ThoughtNode(
            id=root_id,
            thought=problem,
            depth=0,
            state=NodeState.EVALUATED,
            value=0.5,
        )

        tree = ThoughtTree(root_id=root_id)
        tree.add_node(root)
        self._current_tree = tree

        # Add initial thoughts if provided
        if initial_thoughts:
            current = root
            for i, thought in enumerate(initial_thoughts):
                node_id = self._generate_node_id()
                node = ThoughtNode(
                    id=node_id,
                    thought=thought,
                    parent_id=current.id,
                    depth=i + 1,
                    state=NodeState.EVALUATED,
                    value=0.5,
                )
                tree.add_node(node)
                current.children_ids.append(node_id)
                current = node

        logger.info(
            "tot_solve_started",
            problem_length=len(problem),
            strategy=self.config.search_strategy.value,
        )

        # Run search
        if self.config.search_strategy == SearchStrategy.BFS:
            await self._bfs_search(problem, tree)
        elif self.config.search_strategy == SearchStrategy.DFS:
            await self._dfs_search(problem, tree)
        elif self.config.search_strategy == SearchStrategy.BEAM:
            await self._beam_search(problem, tree)
        elif self.config.search_strategy == SearchStrategy.MCTS:
            await self._mcts_search(problem, tree)
        elif self.config.search_strategy == SearchStrategy.BEST_FIRST:
            await self._best_first_search(problem, tree)

        # Find best path
        best_path, best_value = self._find_best_path(tree)
        tree.best_path = best_path
        tree.best_value = best_value

        logger.info(
            "tot_solve_completed",
            best_value=best_value,
            path_length=len(best_path),
            total_nodes=tree.total_nodes,
        )

        return tree.get_path_thoughts(best_path[-1]) if best_path else [], best_value, tree

    async def _generate_thoughts(
        self,
        problem: str,
        path: list[str],
        n: int,
    ) -> list[str]:
        """Generate n possible next thoughts."""
        if not self.generate_fn:
            # Fallback: simple heuristic generation
            return [f"Thought {i + 1}: Consider approach {i + 1}" for i in range(n)]

        path_str = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(path))
        prompt = self.config.thought_prompt_template.format(
            problem=problem,
            path=path_str if path else "No steps yet.",
        )

        thoughts = []
        for _ in range(n):
            try:
                thought = await self.generate_fn(prompt)
                thoughts.append(thought.strip())
            except Exception as e:
                logger.error("thought_generation_error", error=str(e))

        return thoughts

    async def _evaluate_path(
        self,
        problem: str,
        path: list[str],
    ) -> float:
        """Evaluate a reasoning path."""
        if not self.evaluate_fn:
            # Fallback: heuristic evaluation based on path length
            return min(1.0, len(path) * 0.1 + 0.3)

        try:
            value = await self.evaluate_fn(problem, path)
            return max(0.0, min(1.0, value))
        except Exception as e:
            logger.error("path_evaluation_error", error=str(e))
            return 0.5

    async def _bfs_search(self, problem: str, tree: ThoughtTree) -> None:
        """Breadth-first search."""
        queue = [tree.root_id]

        while queue and tree.total_nodes < self.config.max_nodes:
            current_id = queue.pop(0)
            current = tree.get_node(current_id)

            if not current or current.depth >= self.config.max_depth:
                continue

            if current.value >= self.config.early_stop_value:
                break

            # Generate children
            path = tree.get_path_thoughts(current_id)
            thoughts = await self._generate_thoughts(
                problem, path, self.config.branching_factor
            )

            for thought in thoughts:
                child_id = self._generate_node_id()
                child_path = path + [thought]
                value = await self._evaluate_path(problem, child_path)

                child = ThoughtNode(
                    id=child_id,
                    thought=thought,
                    parent_id=current_id,
                    depth=current.depth + 1,
                    state=NodeState.EVALUATED,
                    value=value,
                )

                if value >= self.config.value_threshold:
                    tree.add_node(child)
                    current.children_ids.append(child_id)
                    queue.append(child_id)
                else:
                    child.state = NodeState.PRUNED

    async def _dfs_search(self, problem: str, tree: ThoughtTree) -> None:
        """Depth-first search with backtracking."""
        stack = [tree.root_id]

        while stack and tree.total_nodes < self.config.max_nodes:
            current_id = stack.pop()
            current = tree.get_node(current_id)

            if not current or current.depth >= self.config.max_depth:
                continue

            if current.value >= self.config.early_stop_value:
                break

            # Generate children
            path = tree.get_path_thoughts(current_id)
            thoughts = await self._generate_thoughts(
                problem, path, self.config.branching_factor
            )

            for thought in reversed(thoughts):
                child_id = self._generate_node_id()
                child_path = path + [thought]
                value = await self._evaluate_path(problem, child_path)

                child = ThoughtNode(
                    id=child_id,
                    thought=thought,
                    parent_id=current_id,
                    depth=current.depth + 1,
                    state=NodeState.EVALUATED,
                    value=value,
                )

                if value >= self.config.value_threshold:
                    tree.add_node(child)
                    current.children_ids.append(child_id)
                    stack.append(child_id)
                else:
                    child.state = NodeState.PRUNED

    async def _beam_search(self, problem: str, tree: ThoughtTree) -> None:
        """Beam search keeping top-k paths at each level."""
        beam = [tree.root_id]

        for depth in range(self.config.max_depth):
            if not beam or tree.total_nodes >= self.config.max_nodes:
                break

            all_candidates = []

            for current_id in beam:
                current = tree.get_node(current_id)
                if not current:
                    continue

                if current.value >= self.config.early_stop_value:
                    return

                # Generate children
                path = tree.get_path_thoughts(current_id)
                thoughts = await self._generate_thoughts(
                    problem, path, self.config.branching_factor
                )

                for thought in thoughts:
                    child_id = self._generate_node_id()
                    child_path = path + [thought]
                    value = await self._evaluate_path(problem, child_path)

                    child = ThoughtNode(
                        id=child_id,
                        thought=thought,
                        parent_id=current_id,
                        depth=depth + 1,
                        state=NodeState.EVALUATED,
                        value=value,
                    )

                    if value >= self.config.value_threshold:
                        tree.add_node(child)
                        current.children_ids.append(child_id)
                        all_candidates.append((value, child_id))

            # Keep top beam_width candidates
            all_candidates.sort(reverse=True)
            beam = [cid for _, cid in all_candidates[:self.config.beam_width]]

    async def _mcts_search(self, problem: str, tree: ThoughtTree) -> None:
        """Monte Carlo Tree Search."""
        for _ in range(self.config.num_simulations):
            if tree.total_nodes >= self.config.max_nodes:
                break

            # Selection: traverse tree using UCB
            current_id = tree.root_id
            path = []

            while True:
                current = tree.get_node(current_id)
                if not current:
                    break

                path.append(current_id)

                if current.is_leaf() or current.depth >= self.config.max_depth:
                    break

                # Select child using UCB
                best_child = None
                best_ucb = float("-inf")

                for child_id in current.children_ids:
                    child = tree.get_node(child_id)
                    if child:
                        ucb = self._compute_ucb(child, current.visits)
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_child = child_id

                if best_child:
                    current_id = best_child
                else:
                    break

            current = tree.get_node(current_id)
            if not current:
                continue

            # Expansion: add new child if not terminal
            if current.depth < self.config.max_depth and current.visits > 0:
                thought_path = tree.get_path_thoughts(current_id)
                thoughts = await self._generate_thoughts(problem, thought_path, 1)

                if thoughts:
                    child_id = self._generate_node_id()
                    child = ThoughtNode(
                        id=child_id,
                        thought=thoughts[0],
                        parent_id=current_id,
                        depth=current.depth + 1,
                    )
                    tree.add_node(child)
                    current.children_ids.append(child_id)
                    current_id = child_id
                    path.append(child_id)

            # Simulation: evaluate path
            thought_path = tree.get_path_thoughts(current_id)
            value = await self._evaluate_path(problem, thought_path)

            # Backpropagation: update values along path
            for node_id in path:
                node = tree.get_node(node_id)
                if node:
                    node.visits += 1
                    node.total_reward += value
                    node.value = node.total_reward / node.visits
                    node.state = NodeState.EVALUATED

    async def _best_first_search(self, problem: str, tree: ThoughtTree) -> None:
        """Best-first search using priority queue."""
        # Priority queue: (-value, node_id)
        pq = [(-0.5, tree.root_id)]

        while pq and tree.total_nodes < self.config.max_nodes:
            neg_value, current_id = heapq.heappop(pq)
            current = tree.get_node(current_id)

            if not current or current.depth >= self.config.max_depth:
                continue

            if -neg_value >= self.config.early_stop_value:
                break

            # Generate children
            path = tree.get_path_thoughts(current_id)
            thoughts = await self._generate_thoughts(
                problem, path, self.config.branching_factor
            )

            for thought in thoughts:
                child_id = self._generate_node_id()
                child_path = path + [thought]
                value = await self._evaluate_path(problem, child_path)

                child = ThoughtNode(
                    id=child_id,
                    thought=thought,
                    parent_id=current_id,
                    depth=current.depth + 1,
                    state=NodeState.EVALUATED,
                    value=value,
                )

                if value >= self.config.value_threshold:
                    tree.add_node(child)
                    current.children_ids.append(child_id)
                    heapq.heappush(pq, (-value, child_id))
                else:
                    child.state = NodeState.PRUNED

    def _compute_ucb(self, node: ThoughtNode, parent_visits: int) -> float:
        """Compute UCB score for MCTS."""
        if node.visits == 0:
            return float("inf")

        exploitation = node.total_reward / node.visits
        exploration = self.config.exploration_constant * math.sqrt(
            math.log(parent_visits) / node.visits
        )

        return exploitation + exploration

    def _find_best_path(self, tree: ThoughtTree) -> tuple[list[str], float]:
        """Find the best path in the tree."""
        best_path = []
        best_value = 0.0

        # Find all leaf nodes
        for node in tree.nodes.values():
            if node.is_leaf() and node.state == NodeState.EVALUATED:
                if node.value > best_value:
                    best_value = node.value
                    best_path = [n.id for n in tree.get_path(node.id)]

        return best_path, best_value

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"thought-{self._node_counter}"

    def get_current_tree(self) -> Optional[ThoughtTree]:
        """Get the current tree being explored."""
        return self._current_tree

    def visualize_tree(self, tree: ThoughtTree) -> str:
        """Generate a text visualization of the tree."""
        lines = []

        def _visualize_node(node_id: str, prefix: str = "", is_last: bool = True):
            node = tree.get_node(node_id)
            if not node:
                return

            connector = "└── " if is_last else "├── "
            value_str = f"[{node.value:.2f}]" if node.value else ""

            thought_preview = node.thought[:50] + "..." if len(node.thought) > 50 else node.thought
            lines.append(f"{prefix}{connector}{thought_preview} {value_str}")

            new_prefix = prefix + ("    " if is_last else "│   ")

            for i, child_id in enumerate(node.children_ids):
                _visualize_node(child_id, new_prefix, i == len(node.children_ids) - 1)

        _visualize_node(tree.root_id)
        return "\n".join(lines)
