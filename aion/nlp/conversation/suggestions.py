"""
AION Suggestion Engine - LLM-powered contextual suggestions.

Generates contextual suggestions by combining:
- Static best-practice hints (fast fallback)
- LLM-powered analysis of the actual generated code and spec
- Session context awareness (iteration count, intent history)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import Intent, IntentType, ProgrammingSession
from aion.nlp.utils import parse_json_safe

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class SuggestionEngine:
    """
    Generates proactive, contextual suggestions.

    Uses LLM when available for code-aware suggestions,
    falling back to static best-practice hints.
    """

    def __init__(self, kernel: Optional[AIONKernel] = None):
        self._kernel = kernel

    async def generate(
        self,
        session: ProgrammingSession,
        intent: Optional[Intent] = None,
    ) -> List[str]:
        """Generate contextual suggestions using LLM when available."""
        suggestions: List[str] = []

        # Try LLM-powered suggestions first
        if self._kernel and session.current_code:
            llm_suggestions = await self._llm_suggestions(session, intent)
            suggestions.extend(llm_suggestions)

        # Add static suggestions as fallback or supplement
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

    async def _llm_suggestions(
        self,
        session: ProgrammingSession,
        intent: Optional[Intent] = None,
    ) -> List[str]:
        """Generate LLM-powered contextual suggestions based on actual code."""
        if not self._kernel:
            return []

        code_preview = session.current_code.code[:500] if session.current_code else ""
        spec_info = ""
        if session.current_spec and hasattr(session.current_spec, "to_dict"):
            spec_info = str(session.current_spec.to_dict())[:300]

        intent_type = intent.type.value if intent else "unknown"

        prompt = f"""Given this generated system, suggest 3 specific improvements.

Type: {intent_type}
Iteration: {session.iterations}
Code preview:
```
{code_preview}
```
{f'Spec: {spec_info}' if spec_info else ''}

Return ONLY a JSON array of 3 short suggestion strings (max 100 chars each).
Focus on: security, error handling, performance, usability, testing.
Example: ["Add input validation for the API key parameter", "Consider retry logic for HTTP calls", "Add logging for debugging"]"""

        try:
            response = await self._kernel.llm.complete(
                [{"role": "user", "content": prompt}]
            )
            content = response.content if hasattr(response, "content") else str(response)

            import json
            import re
            # Try parsing JSON array
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    return [str(s)[:150] for s in result[:3]]
            except json.JSONDecodeError:
                pass

            # Try extracting array from text
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    if isinstance(result, list):
                        return [str(s)[:150] for s in result[:3]]
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug("LLM suggestion generation failed", error=str(e))

        return []

    def _intent_suggestions(self, intent: Intent) -> List[str]:
        """Static suggestions based on current intent type (fast fallback)."""
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
