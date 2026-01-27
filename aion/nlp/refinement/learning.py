"""
AION Refinement Learner - Learn from user corrections.

Tracks patterns in user feedback and feeds correction signals
back into the intent classification pipeline to improve accuracy
over time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.nlp.types import IntentType, SpecificationType
from aion.nlp.utils import BoundedList

logger = structlog.get_logger(__name__)


@dataclass
class CorrectionRecord:
    """Record of a user correction."""

    original_intent: str
    corrected_intent: str
    feedback: str
    intent_type: IntentType
    original_type: Optional[IntentType] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RefinementLearner:
    """
    Learns from user corrections to improve system performance.

    Tracks:
    - Common correction patterns (fed back into intent classifier)
    - Frequently misclassified intents with confusion matrix
    - Common missing entities
    - User preference patterns

    The correction data is accessible via `get_intent_bias()` which returns
    scoring adjustments that the intent parser can apply.
    """

    MAX_CORRECTIONS = 10000

    def __init__(self) -> None:
        self._corrections: BoundedList = BoundedList(max_size=self.MAX_CORRECTIONS)
        self._intent_corrections: Counter = Counter()
        self._common_feedback: Counter = Counter()
        self._entity_gaps: Counter = Counter()
        self._user_preferences: Dict[str, Dict[str, Any]] = {}

        # Confusion matrix: tracks (predicted_type -> actual_type) corrections
        self._confusion: Dict[str, Counter] = defaultdict(Counter)

    def record_correction(
        self,
        original: str,
        corrected: str,
        feedback: str,
        intent_type: IntentType,
        original_type: Optional[IntentType] = None,
    ) -> None:
        """Record a user correction."""
        self._corrections.append(CorrectionRecord(
            original_intent=original[:200],
            corrected_intent=corrected[:200],
            feedback=feedback[:200],
            intent_type=intent_type,
            original_type=original_type,
        ))

        # Track patterns
        self._intent_corrections[intent_type.value] += 1
        self._analyze_feedback(feedback)

        # Update confusion matrix for feedback loop
        if original_type and original_type != intent_type:
            self._confusion[original_type.value][intent_type.value] += 1

        logger.debug(
            "Correction recorded",
            intent_type=intent_type.value,
            original_type=original_type.value if original_type else None,
            total_corrections=len(self._corrections),
        )

    def record_entity_gap(self, entity_type: str, context: str) -> None:
        """Record a missing entity that needed clarification."""
        self._entity_gaps[entity_type] += 1

    def record_user_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: Any,
    ) -> None:
        """Record a user preference."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        self._user_preferences[user_id][preference_key] = preference_value

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get known preferences for a user."""
        return self._user_preferences.get(user_id, {})

    def get_common_corrections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common correction patterns."""
        return [
            {"intent_type": k, "count": v}
            for k, v in self._intent_corrections.most_common(limit)
        ]

    def get_common_gaps(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common entity gaps."""
        return [
            {"entity_type": k, "count": v}
            for k, v in self._entity_gaps.most_common(limit)
        ]

    def get_intent_bias(self) -> Dict[str, float]:
        """
        Get scoring bias for intent classification based on learned corrections.

        Returns a dict of intent_type -> bias_score that can be applied
        as an additive adjustment during ensemble scoring.

        Positive bias means the classifier under-predicts this type.
        Negative bias means the classifier over-predicts this type.
        """
        bias: Dict[str, float] = {}
        total = sum(self._intent_corrections.values())
        if total < 5:
            return bias  # Not enough data

        # For each confused pair, boost the correct type
        for predicted_type, corrected_counts in self._confusion.items():
            total_corrections_for_type = sum(corrected_counts.values())
            if total_corrections_for_type >= 3:
                # Penalize the over-predicted type
                bias[predicted_type] = bias.get(predicted_type, 0.0) - 0.05 * min(total_corrections_for_type, 10)
                # Boost the under-predicted types
                for actual_type, count in corrected_counts.most_common(3):
                    bias[actual_type] = bias.get(actual_type, 0.0) + 0.03 * min(count, 10)

        return bias

    def get_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get the full confusion matrix of prediction errors."""
        return {k: dict(v) for k, v in self._confusion.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_corrections": len(self._corrections),
            "top_corrections": self.get_common_corrections(5),
            "top_gaps": self.get_common_gaps(5),
            "users_tracked": len(self._user_preferences),
            "confusion_pairs": sum(sum(v.values()) for v in self._confusion.values()),
            "intent_biases": self.get_intent_bias(),
        }

    def _analyze_feedback(self, feedback: str) -> None:
        """Analyze feedback for common patterns."""
        feedback_lower = feedback.lower()

        # Track common feedback themes
        themes = {
            "naming": ["name", "rename", "call"],
            "parameters": ["parameter", "input", "argument"],
            "triggers": ["trigger", "schedule", "when"],
            "behavior": ["behavior", "should", "instead"],
            "format": ["format", "output", "return"],
        }

        for theme, keywords in themes.items():
            if any(kw in feedback_lower for kw in keywords):
                self._common_feedback[theme] += 1
