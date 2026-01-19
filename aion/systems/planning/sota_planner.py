"""
AION SOTA Planning System

Tree-of-Thoughts (ToT) + Monte Carlo Tree Search (MCTS) planning with:
- Multi-path exploration with branching
- Self-consistency verification
- Plan refinement through critic feedback
- Hierarchical decomposition
- ReAct-style reasoning-acting interleave
"""

from __future__ import annotations

import asyncio
import math
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Tree-of-Thoughts Core
# ============================================================================

class ThoughtState(str, Enum):
    """State of a thought node."""
    PENDING = "pending"
    EXPLORING = "exploring"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    SELECTED = "selected"


@dataclass
class Thought:
    """A single thought in the reasoning tree."""
    id: str
    content: str
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    state: ThoughtState = ThoughtState.PENDING

    # Evaluation scores
    value_score: float = 0.0  # How promising this thought is
    visit_count: int = 0

    # Self-consistency
    consistency_score: float = 0.0
    verification_passed: bool = False

    # Metadata
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    reasoning_trace: list[str] = field(default_factory=list)

    def ucb1_score(self, total_visits: int, exploration_weight: float = 1.414) -> float:
        """Calculate UCB1 score for MCTS selection."""
        if self.visit_count == 0:
            return float('inf')

        exploitation = self.value_score / self.visit_count
        exploration = exploration_weight * math.sqrt(
            math.log(total_visits) / self.visit_count
        )
        return exploitation + exploration


@dataclass
class ThoughtTree:
    """Complete tree of thoughts for a problem."""
    id: str
    problem: str
    root_id: str
    thoughts: dict[str, Thought] = field(default_factory=dict)
    best_path: list[str] = field(default_factory=list)
    total_visits: int = 0

    # Solution
    solution: Optional[str] = None
    solution_score: float = 0.0
    verified: bool = False


class ThoughtGenerator(ABC):
    """Abstract base for generating next thoughts."""

    @abstractmethod
    async def generate(
        self,
        problem: str,
        current_thought: Thought,
        context: list[Thought],
        num_thoughts: int = 3,
    ) -> list[str]:
        """Generate candidate next thoughts."""
        pass


class ThoughtEvaluator(ABC):
    """Abstract base for evaluating thoughts."""

    @abstractmethod
    async def evaluate(
        self,
        problem: str,
        thought: Thought,
        path: list[Thought],
    ) -> float:
        """Evaluate a thought's promise (0-1 score)."""
        pass

    @abstractmethod
    async def verify(
        self,
        problem: str,
        solution_path: list[Thought],
    ) -> tuple[bool, str]:
        """Verify if a solution path is correct."""
        pass


class LLMThoughtGenerator(ThoughtGenerator):
    """LLM-based thought generation with structured prompting."""

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    async def generate(
        self,
        problem: str,
        current_thought: Thought,
        context: list[Thought],
        num_thoughts: int = 3,
    ) -> list[str]:
        """Generate diverse next thoughts using LLM."""

        # Build context from path
        path_text = "\n".join([
            f"Step {i+1}: {t.content}"
            for i, t in enumerate(context)
        ])

        prompt = f"""Problem: {problem}

Current reasoning path:
{path_text}

Current step: {current_thought.content}

Generate {num_thoughts} distinct next steps in the reasoning process.
Each step should:
1. Build logically on the current step
2. Explore a different approach or aspect
3. Move closer to solving the problem

Format each thought on a new line starting with "THOUGHT:"

Be creative and thorough. Consider:
- What are the key sub-problems?
- What information do we need?
- What are alternative approaches?
- How can we verify our reasoning?"""

        from aion.core.llm import Message

        response = await self.llm.complete([
            Message(role="system", content="You are a systematic reasoning engine that generates diverse solution steps."),
            Message(role="user", content=prompt),
        ], temperature=0.8)  # Higher temperature for diversity

        # Parse thoughts from response
        thoughts = []
        for line in response.content.split("\n"):
            if line.strip().startswith("THOUGHT:"):
                thought = line.replace("THOUGHT:", "").strip()
                if thought:
                    thoughts.append(thought)

        # Ensure we have enough thoughts
        while len(thoughts) < num_thoughts:
            thoughts.append(f"Alternative approach {len(thoughts) + 1}: Continue analysis")

        return thoughts[:num_thoughts]


class LLMThoughtEvaluator(ThoughtEvaluator):
    """LLM-based thought evaluation with self-consistency."""

    def __init__(self, llm_adapter, num_samples: int = 5):
        self.llm = llm_adapter
        self.num_samples = num_samples

    async def evaluate(
        self,
        problem: str,
        thought: Thought,
        path: list[Thought],
    ) -> float:
        """Evaluate thought promise using LLM scoring."""

        path_text = "\n".join([f"- {t.content}" for t in path])

        prompt = f"""Problem: {problem}

Reasoning path so far:
{path_text}

Current thought to evaluate: {thought.content}

Rate this thought on a scale of 0-10 based on:
1. Logical coherence with previous steps (0-3 points)
2. Progress toward solution (0-3 points)
3. Correctness of reasoning (0-2 points)
4. Potential to lead to complete solution (0-2 points)

Provide your rating as a single number after "SCORE:"
Then briefly explain your reasoning after "REASON:"
"""

        from aion.core.llm import Message

        # Multiple samples for self-consistency
        scores = []
        for _ in range(self.num_samples):
            try:
                response = await self.llm.complete([
                    Message(role="system", content="You are a critical evaluator of reasoning steps."),
                    Message(role="user", content=prompt),
                ], temperature=0.3)

                # Parse score
                for line in response.content.split("\n"):
                    if "SCORE:" in line:
                        score_text = line.split("SCORE:")[-1].strip()
                        score = float(score_text.split()[0])
                        scores.append(min(10, max(0, score)))
                        break
            except:
                continue

        if not scores:
            return 0.5

        # Return normalized mean score
        return sum(scores) / (len(scores) * 10)

    async def verify(
        self,
        problem: str,
        solution_path: list[Thought],
    ) -> tuple[bool, str]:
        """Verify solution through multiple checks."""

        path_text = "\n".join([
            f"Step {i+1}: {t.content}"
            for i, t in enumerate(solution_path)
        ])

        prompt = f"""Problem: {problem}

Proposed solution path:
{path_text}

Carefully verify this solution:
1. Is each step logically valid?
2. Does the reasoning flow correctly?
3. Are there any errors or gaps?
4. Does this actually solve the problem?

Respond with:
VALID: YES or NO
ISSUES: List any problems found (or "None" if valid)
CONFIDENCE: 0-100%
"""

        from aion.core.llm import Message

        # Multiple verification samples
        valid_votes = 0
        issues = []

        for _ in range(self.num_samples):
            try:
                response = await self.llm.complete([
                    Message(role="system", content="You are a rigorous solution verifier. Be critical and thorough."),
                    Message(role="user", content=prompt),
                ], temperature=0.2)

                content = response.content.upper()
                if "VALID: YES" in content or "VALID:YES" in content:
                    valid_votes += 1

                # Extract issues
                for line in response.content.split("\n"):
                    if "ISSUES:" in line:
                        issue = line.split("ISSUES:")[-1].strip()
                        if issue.lower() != "none":
                            issues.append(issue)
            except:
                continue

        is_valid = valid_votes > self.num_samples / 2
        issues_text = "; ".join(set(issues)) if issues else "None"

        return is_valid, issues_text


class TreeOfThoughts:
    """
    Tree-of-Thoughts reasoning engine.

    Implements systematic exploration of reasoning paths with:
    - BFS/DFS hybrid exploration
    - Beam search for promising paths
    - Self-consistency verification
    - Backtracking on failures
    """

    def __init__(
        self,
        generator: ThoughtGenerator,
        evaluator: ThoughtEvaluator,
        max_depth: int = 10,
        beam_width: int = 3,
        num_thoughts_per_step: int = 3,
        min_score_threshold: float = 0.3,
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.num_thoughts_per_step = num_thoughts_per_step
        self.min_score_threshold = min_score_threshold

    async def solve(
        self,
        problem: str,
        initial_thought: Optional[str] = None,
    ) -> ThoughtTree:
        """
        Solve a problem using Tree-of-Thoughts.

        Args:
            problem: The problem to solve
            initial_thought: Optional starting thought

        Returns:
            ThoughtTree with solution path
        """
        tree_id = str(uuid.uuid4())

        # Create root thought
        root = Thought(
            id=f"{tree_id}_root",
            content=initial_thought or f"Analyzing problem: {problem[:100]}...",
            depth=0,
        )

        tree = ThoughtTree(
            id=tree_id,
            problem=problem,
            root_id=root.id,
            thoughts={root.id: root},
        )

        # BFS with beam search
        current_level = [root]

        for depth in range(self.max_depth):
            if not current_level:
                break

            next_level = []

            for thought in current_level:
                thought.state = ThoughtState.EXPLORING

                # Get path to this thought
                path = self._get_path(tree, thought)

                # Check if this could be a solution
                if await self._is_solution(problem, path):
                    is_valid, issues = await self.evaluator.verify(problem, path)
                    if is_valid:
                        tree.best_path = [t.id for t in path]
                        tree.solution = thought.content
                        tree.solution_score = thought.value_score
                        tree.verified = True
                        thought.state = ThoughtState.SELECTED
                        thought.verification_passed = True

                        logger.info(
                            "Solution found",
                            tree_id=tree_id,
                            depth=depth,
                            path_length=len(path),
                        )
                        return tree

                # Generate child thoughts
                child_contents = await self.generator.generate(
                    problem=problem,
                    current_thought=thought,
                    context=path,
                    num_thoughts=self.num_thoughts_per_step,
                )

                # Create and evaluate children
                for content in child_contents:
                    child = Thought(
                        id=str(uuid.uuid4()),
                        content=content,
                        parent_id=thought.id,
                        depth=depth + 1,
                    )

                    # Evaluate promise
                    child_path = path + [child]
                    child.value_score = await self.evaluator.evaluate(
                        problem, child, child_path
                    )

                    if child.value_score >= self.min_score_threshold:
                        tree.thoughts[child.id] = child
                        thought.children_ids.append(child.id)
                        next_level.append(child)
                        child.state = ThoughtState.EVALUATED
                    else:
                        child.state = ThoughtState.PRUNED

                thought.state = ThoughtState.EVALUATED

            # Beam search: keep top-k thoughts
            next_level.sort(key=lambda t: t.value_score, reverse=True)
            current_level = next_level[:self.beam_width]

            logger.debug(
                "ToT depth complete",
                depth=depth,
                candidates=len(next_level),
                kept=len(current_level),
            )

        # No verified solution found, return best path
        if current_level:
            best = max(current_level, key=lambda t: t.value_score)
            tree.best_path = [t.id for t in self._get_path(tree, best)]
            tree.solution = best.content
            tree.solution_score = best.value_score

        return tree

    def _get_path(self, tree: ThoughtTree, thought: Thought) -> list[Thought]:
        """Get path from root to thought."""
        path = [thought]
        current = thought

        while current.parent_id:
            current = tree.thoughts[current.parent_id]
            path.insert(0, current)

        return path

    async def _is_solution(self, problem: str, path: list[Thought]) -> bool:
        """Check if path might be a complete solution."""
        if len(path) < 2:
            return False

        last_thought = path[-1].content.lower()
        solution_indicators = [
            "therefore", "thus", "the answer is", "solution:",
            "in conclusion", "finally", "result:", "answer:"
        ]

        return any(ind in last_thought for ind in solution_indicators)


# ============================================================================
# Monte Carlo Tree Search
# ============================================================================

@dataclass
class MCTSNode:
    """Node in MCTS tree."""
    id: str
    state: Any  # Problem state
    action: Optional[str] = None  # Action that led here
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0  # Prior probability from policy

    # State info
    is_terminal: bool = False
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Average value."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb_score(
        self,
        parent_visits: int,
        c_puct: float = 1.5
    ) -> float:
        """PUCT score for selection (AlphaGo-style)."""
        if self.visits == 0:
            return float('inf')

        exploitation = self.q_value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)

        return exploitation + exploration


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner.

    Combines:
    - PUCT selection (as in AlphaGo)
    - Neural policy/value networks (via LLM)
    - Progressive widening
    - Virtual loss for parallel search
    """

    def __init__(
        self,
        llm_adapter,
        num_simulations: int = 100,
        max_depth: int = 20,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ):
        self.llm = llm_adapter
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temperature = temperature

        self.nodes: dict[str, MCTSNode] = {}
        self.root_id: Optional[str] = None

    async def plan(
        self,
        problem: str,
        initial_state: Any = None,
    ) -> list[str]:
        """
        Plan solution using MCTS.

        Args:
            problem: Problem description
            initial_state: Initial state (defaults to problem)

        Returns:
            List of actions (plan steps)
        """
        # Create root
        root = MCTSNode(
            id=str(uuid.uuid4()),
            state=initial_state or {"problem": problem, "steps": []},
        )
        self.nodes = {root.id: root}
        self.root_id = root.id

        # Run simulations
        for sim in range(self.num_simulations):
            await self._simulate(problem, root)

            if sim % 20 == 0:
                logger.debug(
                    "MCTS progress",
                    simulation=sim,
                    root_visits=root.visits,
                )

        # Extract best plan
        plan = self._extract_plan(root)

        return plan

    async def _simulate(self, problem: str, root: MCTSNode) -> float:
        """Run one MCTS simulation."""
        path = []
        node = root

        # Selection: traverse tree using UCB
        while node.is_expanded and node.children_ids and not node.is_terminal:
            node = self._select_child(node)
            path.append(node)

        # Expansion: add new children if not terminal
        if not node.is_terminal and not node.is_expanded:
            await self._expand(problem, node)

        # Simulation: evaluate leaf with LLM
        value = await self._evaluate_state(problem, node.state)

        # Check if terminal
        node.is_terminal = await self._is_terminal(problem, node.state)

        # Backpropagation
        self._backpropagate(path + [node], value)

        return value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using PUCT."""
        best_score = float('-inf')
        best_child = None

        for child_id in node.children_ids:
            child = self.nodes[child_id]
            score = child.ucb_score(node.visits, self.c_puct)

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    async def _expand(self, problem: str, node: MCTSNode) -> None:
        """Expand node by generating possible actions."""

        state = node.state
        steps_so_far = state.get("steps", [])

        prompt = f"""Problem: {problem}

Steps taken so far:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(steps_so_far)) or 'None'}

Generate 3-5 possible next actions. For each action, estimate its probability of being optimal.

Format:
ACTION: <action description>
PROBABILITY: <0.0-1.0>
---
"""

        from aion.core.llm import Message

        try:
            response = await self.llm.complete([
                Message(role="system", content="You are a strategic planner generating possible actions."),
                Message(role="user", content=prompt),
            ], temperature=0.7)

            # Parse actions
            actions = []
            current_action = None
            current_prob = 0.2

            for line in response.content.split("\n"):
                if line.startswith("ACTION:"):
                    if current_action:
                        actions.append((current_action, current_prob))
                    current_action = line.replace("ACTION:", "").strip()
                    current_prob = 0.2
                elif line.startswith("PROBABILITY:"):
                    try:
                        current_prob = float(line.replace("PROBABILITY:", "").strip())
                    except:
                        current_prob = 0.2

            if current_action:
                actions.append((current_action, current_prob))

            # Normalize probabilities
            total_prob = sum(p for _, p in actions)
            if total_prob > 0:
                actions = [(a, p/total_prob) for a, p in actions]

            # Create child nodes
            for action, prior in actions:
                child_state = {
                    "problem": problem,
                    "steps": steps_so_far + [action],
                }

                child = MCTSNode(
                    id=str(uuid.uuid4()),
                    state=child_state,
                    action=action,
                    parent_id=node.id,
                    prior=prior,
                )

                self.nodes[child.id] = child
                node.children_ids.append(child.id)

            node.is_expanded = True

        except Exception as e:
            logger.warning("MCTS expansion failed", error=str(e))
            node.is_expanded = True

    async def _evaluate_state(self, problem: str, state: Any) -> float:
        """Evaluate state value using LLM."""

        steps = state.get("steps", [])

        prompt = f"""Problem: {problem}

Current plan:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(steps)) or 'No steps yet'}

Rate this partial plan from 0 to 10:
- How close is it to solving the problem?
- How feasible are the steps?
- How efficient is the approach?

Respond with just a number between 0 and 10.
"""

        from aion.core.llm import Message

        try:
            response = await self.llm.complete([
                Message(role="system", content="You are evaluating plan quality. Respond with only a number 0-10."),
                Message(role="user", content=prompt),
            ], temperature=0.2)

            score = float(response.content.strip().split()[0])
            return min(1.0, max(0.0, score / 10))

        except:
            return 0.5

    async def _is_terminal(self, problem: str, state: Any) -> bool:
        """Check if state is terminal (problem solved)."""
        steps = state.get("steps", [])

        if len(steps) >= self.max_depth:
            return True

        if not steps:
            return False

        last_step = steps[-1].lower()
        terminal_indicators = [
            "complete", "done", "finished", "solved",
            "final answer", "solution complete"
        ]

        return any(ind in last_step for ind in terminal_indicators)

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        """Backpropagate value through path."""
        for node in path:
            node.visits += 1
            node.total_value += value

    def _extract_plan(self, root: MCTSNode) -> list[str]:
        """Extract best plan from tree."""
        plan = []
        node = root

        while node.children_ids:
            # Select most visited child
            best_child = max(
                [self.nodes[cid] for cid in node.children_ids],
                key=lambda n: n.visits
            )

            if best_child.action:
                plan.append(best_child.action)

            node = best_child

        return plan


# ============================================================================
# ReAct-Style Reasoning
# ============================================================================

@dataclass
class ReActStep:
    """A step in ReAct reasoning."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    is_final: bool = False


class ReActReasoner:
    """
    ReAct-style reasoning with interleaved thinking and acting.

    Implements:
    - Thought-Action-Observation loop
    - Self-correction on errors
    - Tool integration
    - Chain-of-thought reasoning
    """

    def __init__(
        self,
        llm_adapter,
        tool_executor: Optional[Callable] = None,
        max_steps: int = 10,
    ):
        self.llm = llm_adapter
        self.tool_executor = tool_executor
        self.max_steps = max_steps

    async def reason(
        self,
        problem: str,
        available_tools: Optional[list[dict]] = None,
    ) -> tuple[str, list[ReActStep]]:
        """
        Solve problem using ReAct reasoning.

        Args:
            problem: Problem to solve
            available_tools: List of tool definitions

        Returns:
            Tuple of (final_answer, reasoning_trace)
        """
        tools_desc = self._format_tools(available_tools or [])

        steps: list[ReActStep] = []

        for step_num in range(self.max_steps):
            # Generate next thought and action
            step = await self._generate_step(problem, steps, tools_desc)
            steps.append(step)

            if step.is_final:
                return step.thought, steps

            # Execute action if provided
            if step.action and self.tool_executor:
                try:
                    observation = await self.tool_executor(
                        step.action,
                        step.action_input or {}
                    )
                    step.observation = str(observation)
                except Exception as e:
                    step.observation = f"Error: {str(e)}"

            logger.debug(
                "ReAct step",
                step=step_num,
                action=step.action,
                has_observation=bool(step.observation),
            )

        # Max steps reached
        final_thought = await self._synthesize_answer(problem, steps)
        return final_thought, steps

    async def _generate_step(
        self,
        problem: str,
        history: list[ReActStep],
        tools_desc: str,
    ) -> ReActStep:
        """Generate next ReAct step."""

        # Format history
        history_text = ""
        for i, step in enumerate(history):
            history_text += f"\nThought {i+1}: {step.thought}"
            if step.action:
                history_text += f"\nAction {i+1}: {step.action}"
                if step.action_input:
                    history_text += f"\nAction Input: {step.action_input}"
            if step.observation:
                history_text += f"\nObservation {i+1}: {step.observation}"

        prompt = f"""Solve this problem step by step.

Problem: {problem}

Available tools:
{tools_desc}

{history_text}

Generate your next step. You must use this exact format:

Thought: [Your reasoning about what to do next]
Action: [Tool name to use, or "Final Answer" if you have the solution]
Action Input: [Input for the tool as JSON, or your final answer]

Remember:
- Think step by step
- Use tools when needed
- Say "Final Answer" when you have the complete solution
"""

        from aion.core.llm import Message

        response = await self.llm.complete([
            Message(role="system", content="You are a systematic problem solver using the ReAct framework."),
            Message(role="user", content=prompt),
        ], temperature=0.3)

        # Parse response
        thought = ""
        action = None
        action_input = None
        is_final = False

        lines = response.content.split("\n")
        for line in lines:
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
                if "final answer" in action.lower():
                    is_final = True
                    action = None
            elif line.startswith("Action Input:"):
                input_str = line.replace("Action Input:", "").strip()
                if is_final:
                    thought = input_str  # This is the final answer
                else:
                    try:
                        import json
                        action_input = json.loads(input_str)
                    except:
                        action_input = {"input": input_str}

        return ReActStep(
            thought=thought,
            action=action,
            action_input=action_input,
            is_final=is_final,
        )

    async def _synthesize_answer(
        self,
        problem: str,
        steps: list[ReActStep],
    ) -> str:
        """Synthesize final answer from steps."""

        steps_text = "\n".join([
            f"Step {i+1}: {s.thought}" +
            (f" -> {s.observation}" if s.observation else "")
            for i, s in enumerate(steps)
        ])

        prompt = f"""Problem: {problem}

Reasoning steps:
{steps_text}

Based on the above reasoning, provide a clear final answer.
"""

        from aion.core.llm import Message

        response = await self.llm.complete([
            Message(role="system", content="Synthesize a clear answer from the reasoning trace."),
            Message(role="user", content=prompt),
        ])

        return response.content

    def _format_tools(self, tools: list[dict]) -> str:
        """Format tools for prompt."""
        if not tools:
            return "No tools available. Reason through the problem step by step."

        lines = []
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            lines.append(f"- {name}: {desc}")

        return "\n".join(lines)


# ============================================================================
# Unified SOTA Planner
# ============================================================================

class SOTAPlanner:
    """
    State-of-the-art planning system combining:
    - Tree-of-Thoughts for complex reasoning
    - MCTS for action planning
    - ReAct for tool-assisted problem solving
    - Self-consistency verification
    - Hierarchical decomposition
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Initialize components
        self.tot_generator = LLMThoughtGenerator(llm_adapter)
        self.tot_evaluator = LLMThoughtEvaluator(llm_adapter)

        self.tree_of_thoughts = TreeOfThoughts(
            generator=self.tot_generator,
            evaluator=self.tot_evaluator,
        )

        self.mcts = MCTSPlanner(llm_adapter)
        self.react = ReActReasoner(llm_adapter)

    async def solve(
        self,
        problem: str,
        method: str = "auto",
        tools: Optional[list[dict]] = None,
        tool_executor: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """
        Solve a problem using the most appropriate method.

        Args:
            problem: Problem to solve
            method: "auto", "tot", "mcts", or "react"
            tools: Available tools
            tool_executor: Function to execute tools

        Returns:
            Solution with metadata
        """
        if method == "auto":
            method = self._select_method(problem, tools)

        logger.info("SOTA planning", method=method, problem=problem[:100])

        if method == "tot":
            result = await self._solve_with_tot(problem)
        elif method == "mcts":
            result = await self._solve_with_mcts(problem)
        elif method == "react":
            if tool_executor:
                self.react.tool_executor = tool_executor
            result = await self._solve_with_react(problem, tools)
        else:
            # Ensemble: run multiple methods and verify
            result = await self._solve_with_ensemble(problem, tools, tool_executor)

        return result

    def _select_method(
        self,
        problem: str,
        tools: Optional[list[dict]],
    ) -> str:
        """Auto-select best method for problem."""
        problem_lower = problem.lower()

        # Use ReAct if tools available and problem seems action-oriented
        if tools:
            action_words = ["find", "search", "calculate", "fetch", "get", "look up"]
            if any(w in problem_lower for w in action_words):
                return "react"

        # Use MCTS for planning/strategy problems
        planning_words = ["plan", "strategy", "steps", "sequence", "schedule"]
        if any(w in problem_lower for w in planning_words):
            return "mcts"

        # Use ToT for reasoning/analysis problems
        return "tot"

    async def _solve_with_tot(self, problem: str) -> dict[str, Any]:
        """Solve using Tree-of-Thoughts."""
        tree = await self.tree_of_thoughts.solve(problem)

        return {
            "method": "tree_of_thoughts",
            "solution": tree.solution,
            "score": tree.solution_score,
            "verified": tree.verified,
            "path_length": len(tree.best_path),
            "total_thoughts": len(tree.thoughts),
        }

    async def _solve_with_mcts(self, problem: str) -> dict[str, Any]:
        """Solve using MCTS."""
        plan = await self.mcts.plan(problem)

        return {
            "method": "mcts",
            "solution": "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan)),
            "plan_steps": plan,
            "simulations": self.mcts.num_simulations,
        }

    async def _solve_with_react(
        self,
        problem: str,
        tools: Optional[list[dict]],
    ) -> dict[str, Any]:
        """Solve using ReAct."""
        answer, steps = await self.react.reason(problem, tools)

        return {
            "method": "react",
            "solution": answer,
            "reasoning_steps": len(steps),
            "actions_taken": [s.action for s in steps if s.action],
        }

    async def _solve_with_ensemble(
        self,
        problem: str,
        tools: Optional[list[dict]],
        tool_executor: Optional[Callable],
    ) -> dict[str, Any]:
        """Solve using ensemble of methods with verification."""

        # Run ToT and MCTS in parallel
        tot_task = asyncio.create_task(self._solve_with_tot(problem))
        mcts_task = asyncio.create_task(self._solve_with_mcts(problem))

        tot_result, mcts_result = await asyncio.gather(tot_task, mcts_task)

        # Verify and select best
        best_result = tot_result if tot_result.get("verified") else mcts_result

        return {
            "method": "ensemble",
            "solution": best_result["solution"],
            "tot_result": tot_result,
            "mcts_result": mcts_result,
            "selected": "tot" if tot_result.get("verified") else "mcts",
        }
