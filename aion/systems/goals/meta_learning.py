"""
AION Goal System - Meta-Learning

SOTA meta-learning for adaptive goal strategies.
The system learns to learn, improving its goal-setting over time.

Key capabilities:
- MAML-style fast adaptation for new goal domains
- Strategy portfolio with adaptive selection
- Hyperparameter self-tuning
- Learning curve prediction
- Transfer learning across goal types
- Curriculum learning for skill acquisition
"""

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum
import numpy as np
from copy import deepcopy
import json
from pathlib import Path

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
)

logger = structlog.get_logger()


@dataclass
class Strategy:
    """A goal pursuit strategy."""

    id: str
    name: str
    description: str

    # Strategy parameters
    parameters: Dict[str, float] = field(default_factory=dict)

    # Performance tracking
    uses: int = 0
    successes: int = 0
    total_reward: float = 0.0
    avg_completion_time: float = 0.0

    # Applicable goal types
    goal_types: List[GoalType] = field(default_factory=list)

    # Meta-parameters for adaptation
    adaptation_rate: float = 0.1
    exploration_bonus: float = 0.1

    def success_rate(self) -> float:
        if self.uses == 0:
            return 0.5  # Prior
        return self.successes / self.uses

    def avg_reward(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.total_reward / self.uses

    def ucb_score(self, total_uses: int) -> float:
        """Upper confidence bound for strategy selection."""
        if self.uses == 0:
            return float('inf')

        exploitation = self.success_rate()
        exploration = self.exploration_bonus * math.sqrt(
            2 * math.log(total_uses + 1) / self.uses
        )
        return exploitation + exploration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "uses": self.uses,
            "successes": self.successes,
            "total_reward": self.total_reward,
            "avg_completion_time": self.avg_completion_time,
            "goal_types": [gt.value for gt in self.goal_types],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            uses=data.get("uses", 0),
            successes=data.get("successes", 0),
            total_reward=data.get("total_reward", 0.0),
            avg_completion_time=data.get("avg_completion_time", 0.0),
            goal_types=[GoalType(gt) for gt in data.get("goal_types", [])],
        )


@dataclass
class TaskEmbedding:
    """Embedding representing a task/goal type for meta-learning."""

    task_id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def distance(self, other: "TaskEmbedding") -> float:
        """Euclidean distance to another embedding."""
        return float(np.linalg.norm(self.vector - other.vector))

    def similarity(self, other: "TaskEmbedding") -> float:
        """Cosine similarity to another embedding."""
        dot = np.dot(self.vector, other.vector)
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


@dataclass
class LearningCurve:
    """Tracks learning progress over time."""

    metric_name: str
    values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    def add_point(self, value: float):
        self.values.append(value)
        self.timestamps.append(datetime.now())

    def current_value(self) -> float:
        if not self.values:
            return 0.0
        return self.values[-1]

    def improvement_rate(self, window: int = 10) -> float:
        """Calculate rate of improvement."""
        if len(self.values) < 2:
            return 0.0

        recent = self.values[-window:]
        if len(recent) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def predict_value(self, steps_ahead: int) -> float:
        """Predict future value based on learning curve."""
        if len(self.values) < 3:
            return self.current_value()

        # Fit power law: y = a * x^b + c (common for learning curves)
        x = np.arange(1, len(self.values) + 1)
        y = np.array(self.values)

        # Simplified: use log-linear fit
        log_x = np.log(x)
        coeffs = np.polyfit(log_x, y, 1)

        future_x = len(self.values) + steps_ahead
        predicted = coeffs[0] * np.log(future_x) + coeffs[1]

        return float(np.clip(predicted, 0, 1))

    def has_plateaued(self, threshold: float = 0.01, window: int = 20) -> bool:
        """Check if learning has plateaued."""
        if len(self.values) < window:
            return False

        recent_improvement = self.improvement_rate(window)
        return abs(recent_improvement) < threshold


class MAMLAdapter:
    """
    Model-Agnostic Meta-Learning (MAML) style adaptation.

    Enables fast adaptation to new goal types with few examples.
    """

    def __init__(
        self,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5,
        param_dim: int = 32,
    ):
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps
        self.param_dim = param_dim

        # Meta-parameters (shared initialization)
        self.meta_params = np.random.randn(param_dim) * 0.1

        # Task-specific adaptations
        self._task_params: Dict[str, np.ndarray] = {}

    def _compute_loss(
        self,
        params: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute loss and gradient for simple linear model."""
        # Simple linear prediction
        pred = np.dot(features, params[:len(features)])
        error = pred - targets

        loss = float(np.mean(error ** 2))
        grad = 2 * np.outer(features, error).mean(axis=1)

        # Pad gradient to param_dim
        if len(grad) < self.param_dim:
            grad = np.pad(grad, (0, self.param_dim - len(grad)))

        return loss, grad

    def adapt(
        self,
        task_id: str,
        support_features: np.ndarray,
        support_targets: np.ndarray,
    ) -> np.ndarray:
        """
        Adapt to a new task using few-shot examples.

        Returns task-specific parameters.
        """
        # Start from meta-parameters
        params = self.meta_params.copy()

        # Inner loop: gradient descent on support set
        for _ in range(self.n_inner_steps):
            loss, grad = self._compute_loss(params, support_features, support_targets)
            params -= self.inner_lr * grad

        self._task_params[task_id] = params
        return params

    def predict(
        self,
        task_id: str,
        features: np.ndarray,
    ) -> float:
        """Predict using task-specific parameters."""
        params = self._task_params.get(task_id, self.meta_params)
        return float(np.dot(features[:len(params)], params[:len(features)]))

    def meta_update(
        self,
        tasks: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ):
        """
        Meta-update using multiple tasks.

        Each task is (task_id, support_features, support_targets, query_features, query_targets).
        """
        meta_grad = np.zeros_like(self.meta_params)

        for task_id, support_X, support_y, query_X, query_y in tasks:
            # Adapt to task
            adapted_params = self.adapt(task_id, support_X, support_y)

            # Compute loss on query set
            loss, grad = self._compute_loss(adapted_params, query_X, query_y)

            # Accumulate meta-gradient
            meta_grad += grad

        # Update meta-parameters
        meta_grad /= len(tasks)
        self.meta_params -= self.outer_lr * meta_grad


class StrategyPortfolio:
    """
    Portfolio of strategies with adaptive selection.

    Uses multi-armed bandit algorithms to select the best strategy.
    """

    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self.total_uses = 0

        # Strategy-goal type affinities
        self._affinities: Dict[Tuple[str, GoalType], float] = {}

        # Initialize default strategies
        self._init_default_strategies()

    def _init_default_strategies(self):
        """Initialize default strategy portfolio."""
        defaults = [
            Strategy(
                id="direct_execution",
                name="Direct Execution",
                description="Execute goal directly without decomposition",
                parameters={"decomposition_threshold": 0.8},
                goal_types=[GoalType.ACHIEVEMENT, GoalType.MAINTENANCE],
            ),
            Strategy(
                id="hierarchical_decomposition",
                name="Hierarchical Decomposition",
                description="Break goal into subgoals recursively",
                parameters={"max_depth": 3, "min_complexity": 0.3},
                goal_types=[GoalType.CREATION, GoalType.OPTIMIZATION],
            ),
            Strategy(
                id="parallel_pursuit",
                name="Parallel Pursuit",
                description="Pursue multiple objectives simultaneously",
                parameters={"max_parallel": 3, "sync_interval": 60},
                goal_types=[GoalType.EXPLORATION, GoalType.LEARNING],
            ),
            Strategy(
                id="iterative_refinement",
                name="Iterative Refinement",
                description="Iteratively improve toward goal",
                parameters={"iterations": 5, "improvement_threshold": 0.1},
                goal_types=[GoalType.OPTIMIZATION],
            ),
            Strategy(
                id="exploratory_search",
                name="Exploratory Search",
                description="Explore solution space before committing",
                parameters={"exploration_budget": 0.3, "beam_width": 5},
                goal_types=[GoalType.EXPLORATION, GoalType.LEARNING],
            ),
        ]

        for strategy in defaults:
            self.strategies[strategy.id] = strategy

    def select_strategy(
        self,
        goal: Goal,
        exploration_rate: float = 0.1,
    ) -> Strategy:
        """Select best strategy for a goal."""
        # Filter applicable strategies
        applicable = [
            s for s in self.strategies.values()
            if not s.goal_types or goal.goal_type in s.goal_types
        ]

        if not applicable:
            applicable = list(self.strategies.values())

        # Epsilon-greedy with UCB
        if random.random() < exploration_rate:
            return random.choice(applicable)

        # UCB selection
        best_strategy = max(applicable, key=lambda s: s.ucb_score(self.total_uses))
        return best_strategy

    def update_strategy(
        self,
        strategy_id: str,
        success: bool,
        reward: float,
        completion_time: float,
    ):
        """Update strategy performance metrics."""
        if strategy_id not in self.strategies:
            return

        strategy = self.strategies[strategy_id]
        strategy.uses += 1
        self.total_uses += 1

        if success:
            strategy.successes += 1

        strategy.total_reward += reward

        # Running average of completion time
        alpha = 0.1
        strategy.avg_completion_time = (
            (1 - alpha) * strategy.avg_completion_time + alpha * completion_time
        )

    def adapt_strategy_params(
        self,
        strategy_id: str,
        feedback: Dict[str, float],
    ):
        """Adapt strategy parameters based on feedback."""
        if strategy_id not in self.strategies:
            return

        strategy = self.strategies[strategy_id]

        for param_name, adjustment in feedback.items():
            if param_name in strategy.parameters:
                old_value = strategy.parameters[param_name]
                new_value = old_value + strategy.adaptation_rate * adjustment
                strategy.parameters[param_name] = new_value

    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """Get strategies ranked by performance."""
        rankings = [
            (s.id, s.success_rate())
            for s in self.strategies.values()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)


class HyperparameterTuner:
    """
    Self-tuning hyperparameters using Bayesian optimization.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_initial: int = 5,
    ):
        self.param_bounds = param_bounds
        self.n_initial = n_initial

        # Observation history
        self._observations: List[Tuple[Dict[str, float], float]] = []

        # Current best
        self._best_params: Optional[Dict[str, float]] = None
        self._best_score: float = float('-inf')

    def suggest(self) -> Dict[str, float]:
        """Suggest next hyperparameter configuration."""
        # Initial random exploration
        if len(self._observations) < self.n_initial:
            return self._random_sample()

        # Bayesian optimization (simplified GP-UCB)
        return self._bayesian_suggest()

    def _random_sample(self) -> Dict[str, float]:
        """Random sample from parameter space."""
        return {
            name: random.uniform(low, high)
            for name, (low, high) in self.param_bounds.items()
        }

    def _bayesian_suggest(self) -> Dict[str, float]:
        """Suggest using Bayesian optimization."""
        # Simplified: combine best found with exploration
        if self._best_params is None:
            return self._random_sample()

        suggestion = {}
        for name, (low, high) in self.param_bounds.items():
            # Perturb best with decay
            best_val = self._best_params.get(name, (low + high) / 2)
            noise_scale = (high - low) * 0.1
            suggested = best_val + random.gauss(0, noise_scale)
            suggestion[name] = np.clip(suggested, low, high)

        return suggestion

    def observe(self, params: Dict[str, float], score: float):
        """Record observation."""
        self._observations.append((params, score))

        if score > self._best_score:
            self._best_score = score
            self._best_params = params.copy()

    def get_best(self) -> Tuple[Dict[str, float], float]:
        """Get best configuration found."""
        return self._best_params or self._random_sample(), self._best_score


class CurriculumLearner:
    """
    Curriculum learning for skill acquisition.

    Automatically sequences goal difficulty for optimal learning.
    """

    def __init__(self, difficulty_increase_rate: float = 0.1):
        self.difficulty_increase_rate = difficulty_increase_rate

        # Skill levels
        self._skill_levels: Dict[str, float] = defaultdict(float)

        # Goal difficulty estimates
        self._difficulty_estimates: Dict[str, float] = {}

        # Learning history
        self._history: List[Tuple[str, float, bool]] = []  # (skill, difficulty, success)

    def estimate_difficulty(self, goal: Goal) -> float:
        """Estimate goal difficulty."""
        if goal.id in self._difficulty_estimates:
            return self._difficulty_estimates[goal.id]

        # Heuristic difficulty estimation
        difficulty = 0.0

        # More criteria = harder
        difficulty += len(goal.success_criteria) * 0.1

        # Longer description = more complex
        difficulty += len(goal.description) / 5000

        # Priority affects perceived difficulty
        priority_diff = {
            GoalPriority.LOW: 0.0,
            GoalPriority.MEDIUM: 0.1,
            GoalPriority.HIGH: 0.2,
            GoalPriority.CRITICAL: 0.3,
        }
        difficulty += priority_diff.get(goal.priority, 0.1)

        # Goal type affects difficulty
        type_diff = {
            GoalType.MAINTENANCE: 0.1,
            GoalType.ACHIEVEMENT: 0.2,
            GoalType.OPTIMIZATION: 0.4,
            GoalType.CREATION: 0.5,
            GoalType.EXPLORATION: 0.3,
            GoalType.LEARNING: 0.3,
        }
        difficulty += type_diff.get(goal.goal_type, 0.2)

        # Constraints add difficulty
        difficulty += len(goal.constraints) * 0.1

        difficulty = min(1.0, difficulty)
        self._difficulty_estimates[goal.id] = difficulty

        return difficulty

    def get_skill_for_goal(self, goal: Goal) -> str:
        """Determine relevant skill for a goal."""
        # Simplified: use goal type as skill
        return goal.goal_type.value

    def get_current_skill_level(self, skill: str) -> float:
        """Get current level for a skill."""
        return self._skill_levels.get(skill, 0.0)

    def is_appropriate_difficulty(self, goal: Goal) -> bool:
        """Check if goal is appropriate for current skill level."""
        difficulty = self.estimate_difficulty(goal)
        skill = self.get_skill_for_goal(goal)
        skill_level = self.get_current_skill_level(skill)

        # Zone of proximal development: slightly above current level
        return skill_level - 0.1 <= difficulty <= skill_level + 0.3

    def update_from_outcome(self, goal: Goal, success: bool):
        """Update skill levels from goal outcome."""
        difficulty = self.estimate_difficulty(goal)
        skill = self.get_skill_for_goal(goal)

        self._history.append((skill, difficulty, success))

        if success:
            # Increase skill level
            current = self._skill_levels[skill]
            increase = self.difficulty_increase_rate * (difficulty - current + 0.5)
            self._skill_levels[skill] = min(1.0, current + max(0, increase))
        else:
            # Small decrease on failure
            current = self._skill_levels[skill]
            self._skill_levels[skill] = max(0.0, current - 0.02)

    def suggest_next_goal(self, candidates: List[Goal]) -> Optional[Goal]:
        """Suggest next goal based on curriculum."""
        # Filter to appropriate difficulty
        appropriate = [g for g in candidates if self.is_appropriate_difficulty(g)]

        if not appropriate:
            # No appropriate goals, pick easiest
            return min(candidates, key=self.estimate_difficulty) if candidates else None

        # Pick goal with highest learning potential
        def learning_potential(g: Goal) -> float:
            difficulty = self.estimate_difficulty(g)
            skill_level = self.get_current_skill_level(self.get_skill_for_goal(g))
            # Optimal difficulty is slightly above current level
            optimal_diff = skill_level + 0.2
            return -abs(difficulty - optimal_diff)

        return max(appropriate, key=learning_potential)

    def get_skill_summary(self) -> Dict[str, float]:
        """Get summary of skill levels."""
        return dict(self._skill_levels)


class TransferLearner:
    """
    Transfer learning across goal types.

    Leverages knowledge from related goals to accelerate learning.
    """

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

        # Task embeddings
        self._embeddings: Dict[str, TaskEmbedding] = {}

        # Transfer weights
        self._transfer_weights: Dict[Tuple[str, str], float] = {}

        # Knowledge base
        self._knowledge: Dict[str, Dict[str, Any]] = {}

    def get_embedding(self, goal: Goal) -> TaskEmbedding:
        """Get or create task embedding for a goal type."""
        task_id = f"{goal.goal_type.value}_{hash(goal.title) % 1000}"

        if task_id not in self._embeddings:
            # Create embedding from goal features
            vector = np.zeros(self.embedding_dim)

            # Encode goal type
            type_idx = hash(goal.goal_type.value) % (self.embedding_dim // 4)
            vector[type_idx] = 1.0

            # Encode title/description features
            text = f"{goal.title} {goal.description}"
            for i, char in enumerate(text[:self.embedding_dim // 2]):
                idx = self.embedding_dim // 4 + (i % (self.embedding_dim // 4))
                vector[idx] += ord(char) / 1000

            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm

            self._embeddings[task_id] = TaskEmbedding(
                task_id=task_id,
                vector=vector,
                metadata={"goal_type": goal.goal_type.value},
            )

        return self._embeddings[task_id]

    def find_similar_tasks(
        self,
        goal: Goal,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find similar tasks for transfer."""
        query_emb = self.get_embedding(goal)

        similarities = []
        for task_id, emb in self._embeddings.items():
            if task_id == query_emb.task_id:
                continue
            sim = query_emb.similarity(emb)
            similarities.append((task_id, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def compute_transfer_weight(
        self,
        source_task: str,
        target_task: str,
    ) -> float:
        """Compute transfer weight between tasks."""
        key = (source_task, target_task)

        if key in self._transfer_weights:
            return self._transfer_weights[key]

        # Based on embedding similarity
        if source_task in self._embeddings and target_task in self._embeddings:
            weight = self._embeddings[source_task].similarity(
                self._embeddings[target_task]
            )
        else:
            weight = 0.0

        self._transfer_weights[key] = weight
        return weight

    def store_knowledge(self, task_id: str, knowledge: Dict[str, Any]):
        """Store learned knowledge for a task."""
        self._knowledge[task_id] = knowledge

    def transfer_knowledge(
        self,
        goal: Goal,
        min_similarity: float = 0.3,
    ) -> Dict[str, Any]:
        """Transfer knowledge from similar tasks."""
        similar = self.find_similar_tasks(goal)
        transferred = {}

        for task_id, similarity in similar:
            if similarity < min_similarity:
                continue

            if task_id not in self._knowledge:
                continue

            source_knowledge = self._knowledge[task_id]

            # Weighted combination of knowledge
            for key, value in source_knowledge.items():
                if key not in transferred:
                    transferred[key] = []

                transferred[key].append({
                    "value": value,
                    "weight": similarity,
                    "source": task_id,
                })

        # Aggregate transferred knowledge
        aggregated = {}
        for key, values in transferred.items():
            if all(isinstance(v["value"], (int, float)) for v in values):
                # Weighted average for numeric values
                total_weight = sum(v["weight"] for v in values)
                aggregated[key] = sum(
                    v["value"] * v["weight"] for v in values
                ) / total_weight if total_weight > 0 else 0
            else:
                # Take highest weight for non-numeric
                aggregated[key] = max(values, key=lambda v: v["weight"])["value"]

        return aggregated


class MetaLearningSystem:
    """
    Complete meta-learning system for AION goals.

    Integrates all meta-learning components for continuous improvement.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else Path("./data/meta_learning")

        # Components
        self.maml = MAMLAdapter()
        self.strategy_portfolio = StrategyPortfolio()
        self.hyperparameter_tuner = HyperparameterTuner({
            "learning_rate": (0.001, 0.1),
            "exploration_rate": (0.05, 0.3),
            "decomposition_threshold": (0.3, 0.9),
            "max_concurrent_goals": (1, 10),
        })
        self.curriculum = CurriculumLearner()
        self.transfer = TransferLearner()

        # Learning curves
        self._learning_curves: Dict[str, LearningCurve] = {
            "success_rate": LearningCurve("success_rate"),
            "avg_completion_time": LearningCurve("avg_completion_time"),
            "strategy_diversity": LearningCurve("strategy_diversity"),
        }

        self._initialized = False

    async def initialize(self):
        """Initialize the meta-learning system."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        await self._load_state()
        self._initialized = True
        logger.info("meta_learning_system_initialized")

    async def shutdown(self):
        """Shutdown the meta-learning system."""
        await self._save_state()
        self._initialized = False
        logger.info("meta_learning_system_shutdown")

    def select_strategy(self, goal: Goal) -> Strategy:
        """Select best strategy for a goal."""
        return self.strategy_portfolio.select_strategy(goal)

    def adapt_to_goal(
        self,
        goal: Goal,
        examples: List[Tuple[Goal, bool]],
    ) -> np.ndarray:
        """Adapt model to a specific goal using few examples."""
        # Convert examples to features/targets
        features = []
        targets = []

        for example_goal, success in examples:
            feat = self._goal_to_features(example_goal)
            features.append(feat)
            targets.append(1.0 if success else 0.0)

        if not features:
            return self.maml.meta_params

        features = np.array(features)
        targets = np.array(targets)

        task_id = f"goal_{goal.goal_type.value}"
        return self.maml.adapt(task_id, features, targets)

    def _goal_to_features(self, goal: Goal) -> np.ndarray:
        """Convert goal to feature vector."""
        features = np.zeros(32)

        # Type encoding
        type_idx = hash(goal.goal_type.value) % 8
        features[type_idx] = 1.0

        # Priority encoding
        priority_idx = 8 + hash(goal.priority.value) % 4
        features[priority_idx] = 1.0

        # Numeric features
        features[12] = len(goal.success_criteria) / 10
        features[13] = len(goal.description) / 1000
        features[14] = len(goal.tags) / 10
        features[15] = len(goal.constraints) / 5

        return features

    def update_from_outcome(
        self,
        goal: Goal,
        strategy_id: str,
        success: bool,
        completion_time: float,
        reward: float,
    ):
        """Update all meta-learning components from outcome."""
        # Update strategy portfolio
        self.strategy_portfolio.update_strategy(
            strategy_id, success, reward, completion_time
        )

        # Update curriculum
        self.curriculum.update_from_outcome(goal, success)

        # Store knowledge for transfer
        task_embedding = self.transfer.get_embedding(goal)
        self.transfer.store_knowledge(task_embedding.task_id, {
            "strategy_used": strategy_id,
            "success": success,
            "completion_time": completion_time,
            "priority": goal.priority.value,
        })

        # Update learning curves
        total_uses = self.strategy_portfolio.total_uses
        if total_uses > 0:
            success_rate = sum(
                s.successes for s in self.strategy_portfolio.strategies.values()
            ) / total_uses
            self._learning_curves["success_rate"].add_point(success_rate)

        # Log progress
        logger.debug(
            "meta_learning_update",
            goal_id=goal.id,
            strategy=strategy_id,
            success=success,
        )

    def suggest_hyperparameters(self) -> Dict[str, float]:
        """Suggest hyperparameters for goal system."""
        return self.hyperparameter_tuner.suggest()

    def observe_hyperparameters(self, params: Dict[str, float], score: float):
        """Record hyperparameter performance."""
        self.hyperparameter_tuner.observe(params, score)

    def suggest_next_goal(self, candidates: List[Goal]) -> Optional[Goal]:
        """Suggest next goal based on curriculum learning."""
        return self.curriculum.suggest_next_goal(candidates)

    def transfer_knowledge(self, goal: Goal) -> Dict[str, Any]:
        """Get transferred knowledge for a goal."""
        return self.transfer.transfer_knowledge(goal)

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status."""
        return {
            "success_rate": {
                "current": self._learning_curves["success_rate"].current_value(),
                "improvement_rate": self._learning_curves["success_rate"].improvement_rate(),
                "plateaued": self._learning_curves["success_rate"].has_plateaued(),
            },
            "skills": self.curriculum.get_skill_summary(),
            "strategy_ranking": self.strategy_portfolio.get_strategy_ranking(),
            "best_hyperparameters": self.hyperparameter_tuner.get_best()[0],
        }

    async def _save_state(self):
        """Save meta-learning state."""
        state = {
            "strategies": {
                sid: s.to_dict()
                for sid, s in self.strategy_portfolio.strategies.items()
            },
            "skills": dict(self.curriculum._skill_levels),
            "learning_curves": {
                name: {
                    "values": curve.values[-1000:],  # Keep recent
                }
                for name, curve in self._learning_curves.items()
            },
        }

        state_file = self.data_dir / "meta_learning_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    async def _load_state(self):
        """Load meta-learning state."""
        state_file = self.data_dir / "meta_learning_state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore strategies
            for sid, sdata in state.get("strategies", {}).items():
                self.strategy_portfolio.strategies[sid] = Strategy.from_dict(sdata)

            # Restore skills
            self.curriculum._skill_levels = defaultdict(
                float, state.get("skills", {})
            )

            # Restore learning curves
            for name, curve_data in state.get("learning_curves", {}).items():
                if name in self._learning_curves:
                    self._learning_curves[name].values = curve_data.get("values", [])

            logger.info("meta_learning_state_loaded")
        except Exception as e:
            logger.warning("meta_learning_state_load_failed", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        return {
            "total_strategy_uses": self.strategy_portfolio.total_uses,
            "strategies": len(self.strategy_portfolio.strategies),
            "skills_tracked": len(self.curriculum._skill_levels),
            "knowledge_entries": len(self.transfer._knowledge),
            "hyperparameter_observations": len(self.hyperparameter_tuner._observations),
            "learning_status": self.get_learning_status(),
        }
