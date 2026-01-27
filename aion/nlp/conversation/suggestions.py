"""
AION Suggestion Engine - Proactive suggestions for users.

Generates contextual suggestions to help users build
better systems through the NLP programming interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from aion.nlp.types import Intent, IntentType, ProgrammingSession

logger = structlog.get_logger(__name__)


class SuggestionEngine:
    """
    Generates proactive suggestions based on session context.

    Suggestions include:
    - Next steps after creation
    - Related systems to build
    - Best practices
    - Common improvements
    """

    def generate(
        self,
        session: ProgrammingSession,
        intent: Optional[Intent] = None,
    ) -> List[str]:
        """Generate contextual suggestions."""
        suggestions: List[str] = []

        if intent:
            suggestions.extend(self._intent_suggestions(intent))

        suggestions.extend(self._session_suggestions(session))

        # Deduplicate and limit
        seen = set()
        unique: List[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique[:5]

    def _intent_suggestions(self, intent: Intent) -> List[str]:
        """Suggestions based on current intent."""
        suggestions: List[str] = []

        if intent.type == IntentType.CREATE_TOOL:
            suggestions.extend([
                "Consider adding error handling for edge cases",
                "You could create a workflow that uses this tool",
                "Consider adding rate limiting if this calls external APIs",
            ])

        elif intent.type == IntentType.CREATE_WORKFLOW:
            suggestions.extend([
                "Consider adding notification steps for failures",
                "You might want to add logging between steps",
                "Think about what should happen if a step fails",
            ])

        elif intent.type == IntentType.CREATE_AGENT:
            suggestions.extend([
                "Consider adding memory to help the agent learn over time",
                "Define clear success criteria for the agent's goals",
                "Think about what tools the agent should have access to",
            ])

        elif intent.type == IntentType.CREATE_API:
            suggestions.extend([
                "Consider adding authentication to protect endpoints",
                "Think about pagination for list endpoints",
                "Consider adding request validation",
            ])

        elif intent.type == IntentType.CREATE_INTEGRATION:
            suggestions.extend([
                "Consider incremental sync to avoid redundant data transfer",
                "Think about conflict resolution strategies",
                "Consider adding a monitoring workflow for sync failures",
            ])

        return suggestions

    def _session_suggestions(self, session: ProgrammingSession) -> List[str]:
        """Suggestions based on session state."""
        suggestions: List[str] = []

        if session.iterations == 0:
            suggestions.append("Describe what you want to build and I'll create it for you")

        elif session.iterations > 3:
            suggestions.append(
                "Multiple iterations detected - consider describing your requirements "
                "more precisely for better results"
            )

        if session.current_code and not session.current_validation:
            suggestions.append("Ready to validate and deploy your system")

        return suggestions
