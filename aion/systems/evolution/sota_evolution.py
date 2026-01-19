"""
AION SOTA Evolution System

Constitutional AI and RLHF-inspired self-improvement with:
- Constitutional principles for safe improvement
- Reward modeling from feedback
- Self-critique and revision
- Online learning from interactions
- Meta-learning for rapid adaptation
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Constitutional AI Principles
# ============================================================================

@dataclass
class ConstitutionalPrinciple:
    """A principle guiding AI behavior."""
    id: str
    name: str
    description: str
    critique_prompt: str
    revision_prompt: str
    priority: int = 1  # Higher = more important
    category: str = "general"


class Constitution:
    """
    Set of principles governing AI behavior.

    Based on Anthropic's Constitutional AI approach.
    """

    DEFAULT_PRINCIPLES = [
        ConstitutionalPrinciple(
            id="helpful",
            name="Helpfulness",
            description="Responses should be helpful and address the user's needs",
            critique_prompt="Does this response actually help the user? Is it relevant and useful?",
            revision_prompt="Revise to be more helpful and directly address the user's needs.",
            priority=3,
            category="utility",
        ),
        ConstitutionalPrinciple(
            id="honest",
            name="Honesty",
            description="Responses should be truthful and not misleading",
            critique_prompt="Is this response truthful? Does it make any false claims or misleading statements?",
            revision_prompt="Revise to be more honest and accurate. Remove or correct any false claims.",
            priority=4,
            category="safety",
        ),
        ConstitutionalPrinciple(
            id="harmless",
            name="Harmlessness",
            description="Responses should not cause harm to users or others",
            critique_prompt="Could this response cause harm? Does it encourage dangerous behavior?",
            revision_prompt="Revise to remove any potentially harmful content while remaining helpful.",
            priority=5,
            category="safety",
        ),
        ConstitutionalPrinciple(
            id="clear",
            name="Clarity",
            description="Responses should be clear and understandable",
            critique_prompt="Is this response clear and easy to understand?",
            revision_prompt="Revise for clarity. Use simpler language and better structure.",
            priority=2,
            category="quality",
        ),
        ConstitutionalPrinciple(
            id="appropriate",
            name="Appropriateness",
            description="Responses should be appropriate for the context",
            critique_prompt="Is this response appropriate for the context and audience?",
            revision_prompt="Revise to be more appropriate for the situation.",
            priority=3,
            category="quality",
        ),
        ConstitutionalPrinciple(
            id="unbiased",
            name="Fairness",
            description="Responses should be fair and unbiased",
            critique_prompt="Does this response show unfair bias? Is it balanced?",
            revision_prompt="Revise to be more balanced and fair, considering multiple perspectives.",
            priority=4,
            category="safety",
        ),
    ]

    def __init__(self, principles: Optional[list[ConstitutionalPrinciple]] = None):
        self.principles = {p.id: p for p in (principles or self.DEFAULT_PRINCIPLES)}

    def get_by_category(self, category: str) -> list[ConstitutionalPrinciple]:
        """Get principles by category."""
        return [p for p in self.principles.values() if p.category == category]

    def get_sorted_by_priority(self) -> list[ConstitutionalPrinciple]:
        """Get principles sorted by priority (highest first)."""
        return sorted(self.principles.values(), key=lambda p: p.priority, reverse=True)


class ConstitutionalCritic:
    """
    Critiques responses based on constitutional principles.
    """

    def __init__(self, llm_adapter, constitution: Constitution):
        self.llm = llm_adapter
        self.constitution = constitution

    async def critique(
        self,
        response: str,
        context: str,
        principles: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        """
        Critique a response against constitutional principles.

        Args:
            response: Response to critique
            context: Context/prompt that led to response
            principles: Specific principles to check (None = all)

        Returns:
            Dict mapping principle_id to critique results
        """
        from aion.core.llm import Message

        if principles:
            to_check = [self.constitution.principles[p] for p in principles if p in self.constitution.principles]
        else:
            to_check = self.constitution.get_sorted_by_priority()

        critiques = {}

        for principle in to_check:
            prompt = f"""Critique this response based on the following principle.

Principle: {principle.name}
Description: {principle.description}

Context/Prompt: {context}

Response to critique: {response}

{principle.critique_prompt}

Provide:
SCORE: 0-10 (10 = perfect adherence to principle)
ISSUES: List specific issues (or "None")
SUGGESTIONS: How to improve
"""

            try:
                result = await self.llm.complete([
                    Message(role="system", content="You are a critical evaluator of AI responses."),
                    Message(role="user", content=prompt),
                ], temperature=0.2)

                # Parse response
                content = result.content
                score = 5  # Default

                import re
                score_match = re.search(r'SCORE:\s*(\d+)', content)
                if score_match:
                    score = int(score_match.group(1))

                issues = []
                issues_match = re.search(r'ISSUES:\s*(.+?)(?=SUGGESTIONS:|$)', content, re.DOTALL)
                if issues_match:
                    issues_text = issues_match.group(1).strip()
                    if issues_text.lower() != "none":
                        issues = [i.strip() for i in issues_text.split('\n') if i.strip()]

                suggestions = ""
                suggestions_match = re.search(r'SUGGESTIONS:\s*(.+)', content, re.DOTALL)
                if suggestions_match:
                    suggestions = suggestions_match.group(1).strip()

                critiques[principle.id] = {
                    "score": score,
                    "issues": issues,
                    "suggestions": suggestions,
                    "passes": score >= 7,
                }

            except Exception as e:
                logger.warning(f"Critique failed for {principle.id}", error=str(e))
                critiques[principle.id] = {"score": 5, "issues": [], "suggestions": "", "passes": True}

        return critiques


class ConstitutionalReviser:
    """
    Revises responses based on constitutional critiques.
    """

    def __init__(self, llm_adapter, constitution: Constitution):
        self.llm = llm_adapter
        self.constitution = constitution

    async def revise(
        self,
        response: str,
        context: str,
        critiques: dict[str, dict],
    ) -> str:
        """
        Revise a response based on critiques.

        Args:
            response: Original response
            context: Original context
            critiques: Critique results

        Returns:
            Revised response
        """
        from aion.core.llm import Message

        # Focus on failing principles
        failing = [
            (pid, crit)
            for pid, crit in critiques.items()
            if not crit.get("passes", True)
        ]

        if not failing:
            return response

        # Build revision prompt
        issues_text = "\n".join([
            f"- {self.constitution.principles[pid].name}: " +
            "; ".join(crit["issues"][:2]) +
            f"\n  Suggestion: {crit['suggestions'][:200]}"
            for pid, crit in failing[:3]
        ])

        prompt = f"""Revise this response to address the identified issues while maintaining helpfulness.

Original context: {context}

Original response: {response}

Issues to address:
{issues_text}

Provide a revised response that fixes these issues while:
1. Remaining helpful and relevant
2. Keeping the core message intact
3. Being concise

Revised response:"""

        try:
            result = await self.llm.complete([
                Message(role="system", content="You revise responses to be better aligned with principles."),
                Message(role="user", content=prompt),
            ], temperature=0.3)

            return result.content

        except:
            return response


# ============================================================================
# Reward Modeling
# ============================================================================

@dataclass
class Preference:
    """A preference between two responses."""
    context: str
    response_a: str
    response_b: str
    preferred: str  # "a", "b", or "tie"
    confidence: float
    feedback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class RewardModel:
    """
    Learns to predict human preferences.

    Uses LLM-based reward modeling inspired by RLHF.
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Preference data
        self.preferences: list[Preference] = []

        # Learned patterns
        self.preference_patterns: dict[str, float] = defaultdict(float)

    def add_preference(self, preference: Preference) -> None:
        """Add a preference example."""
        self.preferences.append(preference)

        # Update patterns (simple heuristic learning)
        preferred_text = preference.response_a if preference.preferred == "a" else preference.response_b
        words = preferred_text.lower().split()

        for word in set(words):
            self.preference_patterns[word] += preference.confidence * 0.01

    async def score(
        self,
        context: str,
        response: str,
    ) -> float:
        """
        Score a response's likely preference.

        Args:
            context: The context/prompt
            response: Response to score

        Returns:
            Score from 0 to 1
        """
        from aion.core.llm import Message

        # Use LLM as reward model
        prompt = f"""Rate this response on how well it addresses the user's needs.

User request: {context}

Response: {response}

Consider:
1. Helpfulness (0-3 points)
2. Clarity (0-2 points)
3. Accuracy (0-3 points)
4. Appropriateness (0-2 points)

Total score (0-10):"""

        try:
            result = await self.llm.complete([
                Message(role="system", content="You rate response quality. End with a single number 0-10."),
                Message(role="user", content=prompt),
            ], temperature=0.1)

            # Parse score
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', result.content.split('\n')[-1])
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score / 10))

        except:
            pass

        # Fallback to pattern matching
        words = response.lower().split()
        pattern_score = sum(self.preference_patterns.get(w, 0) for w in words)
        return min(1.0, max(0.0, 0.5 + pattern_score))

    async def compare(
        self,
        context: str,
        response_a: str,
        response_b: str,
    ) -> tuple[str, float]:
        """
        Compare two responses.

        Returns:
            Tuple of (preferred: "a"/"b", confidence)
        """
        score_a = await self.score(context, response_a)
        score_b = await self.score(context, response_b)

        if abs(score_a - score_b) < 0.1:
            return "tie", 0.5

        preferred = "a" if score_a > score_b else "b"
        confidence = abs(score_a - score_b)

        return preferred, confidence


# ============================================================================
# Self-Critique and Revision Loop
# ============================================================================

class SelfCritiqueLoop:
    """
    Iterative self-critique and revision.

    Implements CAI-style red-teaming of responses.
    """

    def __init__(
        self,
        llm_adapter,
        constitution: Constitution,
        max_iterations: int = 3,
    ):
        self.llm = llm_adapter
        self.constitution = constitution
        self.critic = ConstitutionalCritic(llm_adapter, constitution)
        self.reviser = ConstitutionalReviser(llm_adapter, constitution)
        self.max_iterations = max_iterations

    async def improve_response(
        self,
        context: str,
        initial_response: str,
    ) -> tuple[str, list[dict]]:
        """
        Iteratively improve a response through self-critique.

        Args:
            context: Original context
            initial_response: Initial response to improve

        Returns:
            Tuple of (final_response, improvement_history)
        """
        current_response = initial_response
        history = []

        for iteration in range(self.max_iterations):
            # Critique current response
            critiques = await self.critic.critique(current_response, context)

            # Check if all principles pass
            all_pass = all(c.get("passes", True) for c in critiques.values())

            history.append({
                "iteration": iteration,
                "response": current_response,
                "critiques": critiques,
                "all_pass": all_pass,
            })

            if all_pass:
                break

            # Revise based on critiques
            revised = await self.reviser.revise(current_response, context, critiques)

            if revised == current_response:
                break

            current_response = revised

        return current_response, history


# ============================================================================
# Online Learning from Interactions
# ============================================================================

@dataclass
class Interaction:
    """A recorded interaction for learning."""
    id: str
    context: str
    response: str
    feedback_score: Optional[float] = None
    feedback_text: Optional[str] = None
    was_revised: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class OnlineLearner:
    """
    Learns from ongoing interactions.

    Implements:
    - Feedback collection
    - Pattern extraction
    - Strategy adjustment
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Interaction history
        self.interactions: list[Interaction] = []

        # Learned strategies
        self.positive_patterns: list[str] = []
        self.negative_patterns: list[str] = []

        # Performance tracking
        self.recent_scores: list[float] = []

    def record_interaction(
        self,
        context: str,
        response: str,
        feedback_score: Optional[float] = None,
        feedback_text: Optional[str] = None,
    ) -> str:
        """Record an interaction."""
        interaction = Interaction(
            id=str(uuid.uuid4()),
            context=context,
            response=response,
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        self.interactions.append(interaction)

        if feedback_score is not None:
            self.recent_scores.append(feedback_score)
            if len(self.recent_scores) > 100:
                self.recent_scores = self.recent_scores[-100:]

        return interaction.id

    def add_feedback(
        self,
        interaction_id: str,
        score: float,
        text: Optional[str] = None,
    ) -> None:
        """Add feedback to an interaction."""
        for interaction in self.interactions:
            if interaction.id == interaction_id:
                interaction.feedback_score = score
                interaction.feedback_text = text
                self.recent_scores.append(score)
                break

    async def extract_patterns(self) -> dict[str, list[str]]:
        """Extract patterns from recent interactions."""
        from aion.core.llm import Message

        # Get recent interactions with feedback
        with_feedback = [
            i for i in self.interactions[-50:]
            if i.feedback_score is not None
        ]

        if len(with_feedback) < 5:
            return {"positive": [], "negative": []}

        positive = [i for i in with_feedback if i.feedback_score >= 0.7]
        negative = [i for i in with_feedback if i.feedback_score <= 0.3]

        patterns = {"positive": [], "negative": []}

        if positive:
            positive_examples = "\n\n".join([
                f"Context: {i.context[:100]}\nResponse: {i.response[:200]}"
                for i in positive[:5]
            ])

            prompt = f"""Analyze these highly-rated responses and identify patterns.

Examples:
{positive_examples}

What patterns make these responses good? List 3-5 key patterns."""

            result = await self.llm.complete([
                Message(role="system", content="You identify patterns in successful responses."),
                Message(role="user", content=prompt),
            ])

            patterns["positive"] = [
                line.strip() for line in result.content.split('\n')
                if line.strip() and not line.startswith('#')
            ][:5]

            self.positive_patterns = patterns["positive"]

        if negative:
            negative_examples = "\n\n".join([
                f"Context: {i.context[:100]}\nResponse: {i.response[:200]}"
                for i in negative[:5]
            ])

            prompt = f"""Analyze these poorly-rated responses and identify what went wrong.

Examples:
{negative_examples}

What patterns make these responses bad? List 3-5 key issues to avoid."""

            result = await self.llm.complete([
                Message(role="system", content="You identify patterns in unsuccessful responses."),
                Message(role="user", content=prompt),
            ])

            patterns["negative"] = [
                line.strip() for line in result.content.split('\n')
                if line.strip() and not line.startswith('#')
            ][:5]

            self.negative_patterns = patterns["negative"]

        return patterns

    def get_performance_trend(self) -> dict[str, float]:
        """Get performance metrics."""
        if not self.recent_scores:
            return {"average": 0.5, "trend": 0.0}

        average = sum(self.recent_scores) / len(self.recent_scores)

        # Calculate trend (last 20 vs previous 20)
        if len(self.recent_scores) >= 40:
            recent = self.recent_scores[-20:]
            previous = self.recent_scores[-40:-20]
            trend = (sum(recent) / len(recent)) - (sum(previous) / len(previous))
        else:
            trend = 0.0

        return {"average": average, "trend": trend}


# ============================================================================
# Meta-Learning for Rapid Adaptation
# ============================================================================

class MetaLearner:
    """
    Meta-learning for rapid adaptation to new tasks.

    Implements MAML-inspired few-shot learning.
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Task embeddings
        self.task_strategies: dict[str, dict] = {}

        # Few-shot examples per task type
        self.task_examples: dict[str, list[dict]] = defaultdict(list)

    async def adapt_to_task(
        self,
        task_description: str,
        examples: list[tuple[str, str]],  # (input, output) pairs
    ) -> str:
        """
        Rapidly adapt to a new task from few examples.

        Args:
            task_description: Description of the task
            examples: Few-shot examples

        Returns:
            Task strategy ID
        """
        task_id = str(uuid.uuid4())

        # Extract strategy from examples
        from aion.core.llm import Message

        examples_text = "\n\n".join([
            f"Input: {inp}\nOutput: {out}"
            for inp, out in examples
        ])

        prompt = f"""Learn the pattern from these examples for a new task.

Task: {task_description}

Examples:
{examples_text}

Describe the strategy/pattern used to transform inputs to outputs.
Be specific about:
1. What the task is
2. Key transformations applied
3. Output format and style
"""

        result = await self.llm.complete([
            Message(role="system", content="You learn task strategies from examples."),
            Message(role="user", content=prompt),
        ])

        self.task_strategies[task_id] = {
            "description": task_description,
            "strategy": result.content,
            "examples": examples,
        }

        self.task_examples[task_description] = [
            {"input": inp, "output": out}
            for inp, out in examples
        ]

        return task_id

    async def apply_strategy(
        self,
        task_id: str,
        new_input: str,
    ) -> str:
        """
        Apply a learned strategy to new input.

        Args:
            task_id: Task strategy ID
            new_input: New input to process

        Returns:
            Generated output
        """
        if task_id not in self.task_strategies:
            return ""

        strategy = self.task_strategies[task_id]

        from aion.core.llm import Message

        examples_text = "\n\n".join([
            f"Input: {ex[0]}\nOutput: {ex[1]}"
            for ex in strategy["examples"]
        ])

        prompt = f"""Apply the learned strategy to a new input.

Task: {strategy['description']}

Strategy learned:
{strategy['strategy']}

Examples:
{examples_text}

New input: {new_input}

Apply the same strategy to produce the output:"""

        result = await self.llm.complete([
            Message(role="system", content="You apply learned strategies consistently."),
            Message(role="user", content=prompt),
        ])

        return result.content


# ============================================================================
# SOTA Evolution Engine
# ============================================================================

class SOTAEvolutionEngine:
    """
    State-of-the-art self-improvement combining:
    - Constitutional AI principles
    - Reward modeling from feedback
    - Self-critique and revision
    - Online learning
    - Meta-learning for adaptation
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Components
        self.constitution = Constitution()
        self.critic = ConstitutionalCritic(llm_adapter, self.constitution)
        self.reviser = ConstitutionalReviser(llm_adapter, self.constitution)
        self.reward_model = RewardModel(llm_adapter)
        self.self_critique = SelfCritiqueLoop(llm_adapter, self.constitution)
        self.online_learner = OnlineLearner(llm_adapter)
        self.meta_learner = MetaLearner(llm_adapter)

        # State
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the evolution engine."""
        logger.info("Initializing SOTA Evolution Engine")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        self._initialized = False

    async def improve_response(
        self,
        context: str,
        response: str,
        max_iterations: int = 3,
    ) -> tuple[str, dict]:
        """
        Improve a response using constitutional AI.

        Args:
            context: Original context
            response: Response to improve
            max_iterations: Max improvement iterations

        Returns:
            Tuple of (improved_response, improvement_details)
        """
        improved, history = await self.self_critique.improve_response(
            context, response
        )

        details = {
            "iterations": len(history),
            "original_critiques": history[0]["critiques"] if history else {},
            "final_critiques": history[-1]["critiques"] if history else {},
            "improved": improved != response,
        }

        return improved, details

    async def score_response(
        self,
        context: str,
        response: str,
    ) -> float:
        """Score a response using reward model."""
        return await self.reward_model.score(context, response)

    async def select_best_response(
        self,
        context: str,
        responses: list[str],
    ) -> tuple[str, int]:
        """
        Select the best response from candidates.

        Returns:
            Tuple of (best_response, index)
        """
        scores = []
        for response in responses:
            score = await self.reward_model.score(context, response)
            scores.append(score)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return responses[best_idx], best_idx

    def record_feedback(
        self,
        context: str,
        response: str,
        score: float,
        feedback_text: Optional[str] = None,
    ) -> str:
        """Record feedback for learning."""
        return self.online_learner.record_interaction(
            context, response, score, feedback_text
        )

    async def learn_from_feedback(self) -> dict:
        """Extract patterns from recent feedback."""
        return await self.online_learner.extract_patterns()

    async def adapt_to_task(
        self,
        task: str,
        examples: list[tuple[str, str]],
    ) -> str:
        """Adapt to a new task from examples."""
        return await self.meta_learner.adapt_to_task(task, examples)

    async def apply_task_strategy(
        self,
        task_id: str,
        input_text: str,
    ) -> str:
        """Apply learned task strategy."""
        return await self.meta_learner.apply_strategy(task_id, input_text)

    def get_stats(self) -> dict[str, Any]:
        """Get evolution statistics."""
        performance = self.online_learner.get_performance_trend()

        return {
            "total_interactions": len(self.online_learner.interactions),
            "preferences_recorded": len(self.reward_model.preferences),
            "task_strategies": len(self.meta_learner.task_strategies),
            "performance_average": performance["average"],
            "performance_trend": performance["trend"],
            "positive_patterns": self.online_learner.positive_patterns,
            "negative_patterns": self.online_learner.negative_patterns,
        }
