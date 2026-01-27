"""
AION Clarification Engine - Resolve ambiguity in user requests.

Generates targeted clarification questions and processes
user responses to refine intent understanding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import Entity, EntityType, Intent, IntentType

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class ClarificationEngine:
    """
    Generates and resolves clarification questions.

    Uses a priority-based question system that asks the most
    important questions first to minimize user friction.
    """

    def __init__(self, kernel: AIONKernel):
        self.kernel = kernel

    async def generate_questions(self, intent: Intent) -> List[str]:
        """Generate prioritized clarification questions for an intent."""
        questions: List[str] = []

        # Critical questions (must-have info)
        questions.extend(self._critical_questions(intent))

        # Important questions (needed for quality output)
        questions.extend(self._important_questions(intent))

        # Nice-to-have questions (for polish)
        if intent.confidence < 0.5:
            questions.extend(self._disambiguation_questions(intent))

        # Deduplicate while preserving order
        seen = set()
        unique: List[str] = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique[:5]  # Limit to 5 questions max

    async def apply_response(
        self,
        intent: Intent,
        question: str,
        response: str,
    ) -> Intent:
        """Apply a user's clarification response to the intent."""

        # Use LLM to interpret the response in context
        prompt = f"""Given this context:
- Original request: "{intent.raw_input}"
- Clarification question: "{question}"
- User response: "{response}"
- Current intent type: {intent.type.value}

Extract any new information from the user's response.
Respond with ONLY valid JSON:
{{
    "name": "<updated name or null>",
    "description": "<updated description or null>",
    "parameters": [
        {{"name": "<param>", "type": "<type>", "description": "<desc>"}}
    ],
    "trigger": "<trigger description or null>",
    "additional_context": "<any other info>"
}}"""

        try:
            llm_response = await self.kernel.llm.complete(
                [{"role": "user", "content": prompt}]
            )
            content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

            import json
            data = json.loads(content) if content.strip().startswith("{") else {}

            # Merge extracted information
            if data.get("name"):
                intent.parameters["name"] = data["name"]
            if data.get("description"):
                intent.parameters["description"] = data["description"]
            if data.get("parameters"):
                existing = intent.parameters.get("inputs", [])
                existing.extend(data["parameters"])
                intent.parameters["inputs"] = existing
            if data.get("trigger"):
                triggers = intent.parameters.get("triggers", [])
                triggers.append(data["trigger"])
                intent.parameters["triggers"] = triggers

        except Exception as e:
            logger.warning("Failed to process clarification response", error=str(e))
            # Fallback: just add as description context
            existing_desc = intent.parameters.get("description", "")
            intent.parameters["description"] = f"{existing_desc} {response}".strip()

        # Re-check if still needs clarification
        remaining = [q for q in intent.clarification_questions if q != question]
        intent.clarification_questions = remaining
        intent.needs_clarification = len(remaining) > 0

        return intent

    def _critical_questions(self, intent: Intent) -> List[str]:
        """Questions about must-have information."""
        questions: List[str] = []

        if intent.type.is_creation:
            if not intent.name and not intent.parameters.get("name"):
                type_label = intent.type.value.replace("create_", "")
                questions.append(f"What would you like to name this {type_label}?")

        if intent.type == IntentType.CREATE_WORKFLOW:
            if not intent.get_entities(EntityType.TRIGGER) and not intent.parameters.get("triggers"):
                questions.append(
                    "What should trigger this workflow? Options include: "
                    "schedule (e.g., every hour), event (e.g., new file), "
                    "webhook, or manual trigger."
                )

        if intent.type == IntentType.CREATE_INTEGRATION:
            if not intent.get_entities(EntityType.DATA_SOURCE):
                questions.append("What system should data come FROM?")
            if not intent.get_entities(EntityType.DATA_TARGET):
                questions.append("What system should data go TO?")

        return questions

    def _important_questions(self, intent: Intent) -> List[str]:
        """Questions for quality output."""
        questions: List[str] = []

        if intent.type == IntentType.CREATE_TOOL:
            if not intent.parameters.get("inputs"):
                questions.append("What inputs (parameters) should this tool accept?")

        if intent.type == IntentType.CREATE_AGENT:
            if not intent.parameters.get("constraints"):
                questions.append(
                    "Are there any constraints or limitations for this agent? "
                    "(e.g., read-only access, specific data only)"
                )

        if intent.type == IntentType.CREATE_API:
            if not intent.parameters.get("outputs"):
                questions.append("What data model or resources should this API manage?")

        return questions

    def _disambiguation_questions(self, intent: Intent) -> List[str]:
        """Questions to resolve ambiguity."""
        questions: List[str] = []

        if intent.confidence < 0.5:
            # Very low confidence - ask about the intent itself
            type_options = []
            for pred in intent.predictions:
                if pred.get("confidence", 0) > 0.2:
                    type_options.append(pred.get("type", ""))

            if len(type_options) > 1:
                options_str = " or ".join(
                    t.replace("create_", "").replace("_", " ")
                    for t in type_options[:3]
                )
                questions.append(
                    f"I want to make sure I understand correctly. "
                    f"Would you like to create a {options_str}?"
                )

        return questions
