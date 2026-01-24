"""
Meta-Learning System

Implements learning-to-learn capabilities for agents,
enabling rapid adaptation to new tasks.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class TaskDistribution:
    """A distribution of similar tasks."""

    id: str
    name: str
    description: str
    task_features: dict[str, Any] = field(default_factory=dict)
    sample_tasks: list[dict[str, Any]] = field(default_factory=list)
    optimal_strategy: Optional[str] = None
    avg_performance: float = 0.0
    task_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_features": self.task_features,
            "sample_count": len(self.sample_tasks),
            "optimal_strategy": self.optimal_strategy,
            "avg_performance": self.avg_performance,
            "task_count": self.task_count,
        }


@dataclass
class LearningCurve:
    """Tracks learning progress on a task distribution."""

    distribution_id: str
    performances: list[float] = field(default_factory=list)
    strategies_tried: list[str] = field(default_factory=list)
    adaptation_steps: int = 0
    converged: bool = False
    convergence_threshold: float = 0.9

    def add_performance(self, performance: float, strategy: str) -> None:
        """Add a performance measurement."""
        self.performances.append(performance)
        self.strategies_tried.append(strategy)
        self.adaptation_steps += 1

        # Check convergence
        if len(self.performances) >= 5:
            recent = self.performances[-5:]
            if min(recent) >= self.convergence_threshold:
                self.converged = True

    @property
    def current_performance(self) -> float:
        """Get current performance level."""
        return self.performances[-1] if self.performances else 0.0

    @property
    def learning_rate(self) -> float:
        """Calculate learning rate (improvement per step)."""
        if len(self.performances) < 2:
            return 0.0

        improvements = [
            self.performances[i] - self.performances[i - 1]
            for i in range(1, len(self.performances))
        ]
        return sum(improvements) / len(improvements)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "distribution_id": self.distribution_id,
            "performances": self.performances,
            "adaptation_steps": self.adaptation_steps,
            "current_performance": self.current_performance,
            "learning_rate": self.learning_rate,
            "converged": self.converged,
        }


@dataclass
class MetaKnowledge:
    """Knowledge about how to learn specific task types."""

    task_type: str
    best_initial_strategy: str
    effective_adaptations: list[str] = field(default_factory=list)
    avg_steps_to_converge: float = 0.0
    success_rate: float = 0.0
    sample_size: int = 0

    def update(self, steps: int, success: bool, strategy: str) -> None:
        """Update meta-knowledge with new experience."""
        self.sample_size += 1

        # Update average steps
        alpha = 1.0 / self.sample_size
        self.avg_steps_to_converge = (
            (1 - alpha) * self.avg_steps_to_converge + alpha * steps
        )

        # Update success rate
        self.success_rate = (
            (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        )

        # Track effective adaptations
        if success and strategy not in self.effective_adaptations:
            self.effective_adaptations.append(strategy)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "best_initial_strategy": self.best_initial_strategy,
            "effective_adaptations": self.effective_adaptations,
            "avg_steps_to_converge": self.avg_steps_to_converge,
            "success_rate": self.success_rate,
            "sample_size": self.sample_size,
        }


class MetaLearner:
    """
    Meta-learning system for learning-to-learn.

    Features:
    - Task distribution learning
    - Fast adaptation to new tasks
    - Strategy transfer across tasks
    - Learning curve analysis
    - Meta-knowledge accumulation
    """

    def __init__(
        self,
        agent_id: str,
        initial_strategies: Optional[list[str]] = None,
    ):
        self.agent_id = agent_id
        self.strategies = initial_strategies or [
            "direct",
            "decompose",
            "analogize",
            "search",
            "reason",
            "verify",
        ]

        # Storage
        self._distributions: dict[str, TaskDistribution] = {}
        self._learning_curves: dict[str, LearningCurve] = {}
        self._meta_knowledge: dict[str, MetaKnowledge] = {}

        # Current learning state
        self._current_distribution: Optional[str] = None
        self._current_strategy: Optional[str] = None

        # Statistics
        self._total_tasks = 0
        self._fast_adaptations = 0  # Tasks solved in <=3 steps
        self._distribution_counter = 0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize meta-learner."""
        self._initialized = True
        logger.info("meta_learner_initialized", agent_id=self.agent_id)

    async def shutdown(self) -> None:
        """Shutdown meta-learner."""
        self._initialized = False
        logger.info("meta_learner_shutdown")

    async def identify_task_distribution(
        self,
        task_description: str,
        task_features: Optional[dict[str, Any]] = None,
    ) -> TaskDistribution:
        """Identify which task distribution a new task belongs to."""
        features = task_features or self._extract_features(task_description)

        # Find matching distribution
        best_match = None
        best_score = 0.0

        for dist in self._distributions.values():
            score = self._compute_similarity(features, dist.task_features)
            if score > best_score:
                best_score = score
                best_match = dist

        # Create new distribution if no good match
        if best_match is None or best_score < 0.5:
            self._distribution_counter += 1
            new_dist = TaskDistribution(
                id=f"dist-{self._distribution_counter}",
                name=f"Task type {self._distribution_counter}",
                description=task_description[:100],
                task_features=features,
            )
            self._distributions[new_dist.id] = new_dist
            self._learning_curves[new_dist.id] = LearningCurve(
                distribution_id=new_dist.id
            )
            best_match = new_dist

        best_match.task_count += 1
        best_match.sample_tasks.append({
            "description": task_description[:200],
            "features": features,
        })

        self._current_distribution = best_match.id

        return best_match

    async def select_initial_strategy(
        self,
        distribution: TaskDistribution,
    ) -> str:
        """Select the best initial strategy for a task distribution."""
        # Check meta-knowledge
        task_type = distribution.task_features.get("type", "unknown")

        if task_type in self._meta_knowledge:
            meta = self._meta_knowledge[task_type]
            if meta.success_rate > 0.7:
                self._current_strategy = meta.best_initial_strategy
                return meta.best_initial_strategy

        # Use distribution's optimal strategy if known
        if distribution.optimal_strategy:
            self._current_strategy = distribution.optimal_strategy
            return distribution.optimal_strategy

        # Fall back to most generally successful strategy
        strategy_scores = defaultdict(float)
        for meta in self._meta_knowledge.values():
            strategy_scores[meta.best_initial_strategy] += meta.success_rate

        if strategy_scores:
            best = max(strategy_scores, key=strategy_scores.get)
            self._current_strategy = best
            return best

        # Default strategy
        self._current_strategy = "reason"
        return "reason"

    async def adapt_strategy(
        self,
        performance: float,
        feedback: Optional[str] = None,
    ) -> str:
        """Adapt strategy based on performance."""
        if not self._current_distribution:
            return self._current_strategy or "reason"

        # Record performance
        curve = self._learning_curves.get(self._current_distribution)
        if curve:
            curve.add_performance(performance, self._current_strategy or "unknown")

        # If performing well, stick with current strategy
        if performance >= 0.8:
            return self._current_strategy or "reason"

        # Try a different strategy
        tried = set(curve.strategies_tried if curve else [])
        untried = [s for s in self.strategies if s not in tried]

        if untried:
            self._current_strategy = untried[0]
        else:
            # Cycle back to best performing
            if curve and curve.performances:
                best_idx = curve.performances.index(max(curve.performances))
                self._current_strategy = curve.strategies_tried[best_idx]

        return self._current_strategy or "reason"

    async def complete_task(
        self,
        success: bool,
        final_performance: float,
        steps_taken: int,
    ) -> None:
        """Record task completion and update meta-knowledge."""
        self._total_tasks += 1

        if steps_taken <= 3 and success:
            self._fast_adaptations += 1

        if not self._current_distribution:
            return

        dist = self._distributions.get(self._current_distribution)
        curve = self._learning_curves.get(self._current_distribution)

        if dist:
            # Update distribution
            alpha = 1.0 / dist.task_count
            dist.avg_performance = (
                (1 - alpha) * dist.avg_performance + alpha * final_performance
            )

            if success and self._current_strategy:
                dist.optimal_strategy = self._current_strategy

        # Update meta-knowledge
        task_type = dist.task_features.get("type", "unknown") if dist else "unknown"

        if task_type not in self._meta_knowledge:
            self._meta_knowledge[task_type] = MetaKnowledge(
                task_type=task_type,
                best_initial_strategy=self._current_strategy or "reason",
            )

        meta = self._meta_knowledge[task_type]
        meta.update(steps_taken, success, self._current_strategy or "reason")

        logger.info(
            "task_completed",
            success=success,
            steps=steps_taken,
            distribution=self._current_distribution,
        )

        # Reset current state
        self._current_distribution = None
        self._current_strategy = None

    def _extract_features(self, task_description: str) -> dict[str, Any]:
        """Extract features from task description."""
        desc_lower = task_description.lower()

        # Simple feature extraction
        features = {
            "length": len(task_description),
            "type": "unknown",
            "complexity": "medium",
        }

        # Detect task type
        type_keywords = {
            "research": ["research", "find", "search", "investigate"],
            "code": ["code", "implement", "program", "function", "write code"],
            "analysis": ["analyze", "analysis", "evaluate", "assess"],
            "writing": ["write", "compose", "draft", "create content"],
            "planning": ["plan", "strategy", "schedule", "organize"],
        }

        for task_type, keywords in type_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                features["type"] = task_type
                break

        # Detect complexity
        if len(task_description) > 500 or "complex" in desc_lower:
            features["complexity"] = "high"
        elif len(task_description) < 100 or "simple" in desc_lower:
            features["complexity"] = "low"

        return features

    def _compute_similarity(
        self,
        features1: dict[str, Any],
        features2: dict[str, Any],
    ) -> float:
        """Compute similarity between feature sets."""
        if not features1 or not features2:
            return 0.0

        matches = 0
        total = 0

        for key in set(features1.keys()) | set(features2.keys()):
            if key in features1 and key in features2:
                total += 1
                if features1[key] == features2[key]:
                    matches += 1

        return matches / total if total > 0 else 0.0

    def get_learning_curve(
        self,
        distribution_id: str,
    ) -> Optional[LearningCurve]:
        """Get learning curve for a distribution."""
        return self._learning_curves.get(distribution_id)

    def get_meta_knowledge(self, task_type: str) -> Optional[MetaKnowledge]:
        """Get meta-knowledge for a task type."""
        return self._meta_knowledge.get(task_type)

    def get_stats(self) -> dict[str, Any]:
        """Get meta-learning statistics."""
        return {
            "agent_id": self.agent_id,
            "total_tasks": self._total_tasks,
            "fast_adaptations": self._fast_adaptations,
            "fast_adaptation_rate": self._fast_adaptations / max(1, self._total_tasks),
            "distributions": len(self._distributions),
            "meta_knowledge_entries": len(self._meta_knowledge),
            "distribution_summary": [
                {
                    "id": d.id,
                    "name": d.name,
                    "task_count": d.task_count,
                    "avg_performance": d.avg_performance,
                }
                for d in list(self._distributions.values())[:5]
            ],
            "meta_knowledge_summary": [
                {
                    "type": m.task_type,
                    "strategy": m.best_initial_strategy,
                    "success_rate": m.success_rate,
                }
                for m in self._meta_knowledge.values()
            ],
        }
