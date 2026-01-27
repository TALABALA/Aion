"""
AION Refinement Learner - Learn from user corrections.

Tracks patterns in user feedback to improve future
intent parsing and code generation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.nlp.types import IntentType, SpecificationType

logger = structlog.get_logger(__name__)


@dataclass
class CorrectionRecord:
    """Record of a user correction."""

    original_intent: str
    corrected_intent: str
    feedback: str
    intent_type: IntentType
    timestamp: datetime = field(default_factory=datetime.now)


class RefinementLearner:
    """
    Learns from user corrections to improve system performance.

    Tracks:
    - Common correction patterns
    - Frequently misclassified intents
    - Common missing entities
    - User preference patterns
    """

    def __init__(self) -> None:
        self._corrections: List[CorrectionRecord] = []
        self._intent_corrections: Counter = Counter()
        self._common_feedback: Counter = Counter()
        self._entity_gaps: Counter = Counter()
        self._user_preferences: Dict[str, Dict[str, Any]] = {}

    def record_correction(
        self,
        original: str,
        corrected: str,
        feedback: str,
        intent_type: IntentType,
    ) -> None:
        """Record a user correction."""
        self._corrections.append(CorrectionRecord(
            original_intent=original[:200],
            corrected_intent=corrected[:200],
            feedback=feedback[:200],
            intent_type=intent_type,
        ))

        # Track patterns
        self._intent_corrections[intent_type.value] += 1
        self._analyze_feedback(feedback)

        logger.debug(
            "Correction recorded",
            intent_type=intent_type.value,
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

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_corrections": len(self._corrections),
            "top_corrections": self.get_common_corrections(5),
            "top_gaps": self.get_common_gaps(5),
            "users_tracked": len(self._user_preferences),
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
