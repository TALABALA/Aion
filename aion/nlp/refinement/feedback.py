"""
AION Feedback Processor - Process user feedback for iterative refinement.

Interprets user feedback to modify specifications and regenerate code.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import GeneratedCode, Intent, ProgrammingSession, Specification

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class FeedbackType(str, Enum):
    """Classification of feedback types."""

    ADD_FEATURE = "add_feature"
    REMOVE_FEATURE = "remove_feature"
    CHANGE_BEHAVIOR = "change_behavior"
    FIX_BUG = "fix_bug"
    CHANGE_NAME = "change_name"
    ADD_PARAMETER = "add_parameter"
    CHANGE_TRIGGER = "change_trigger"
    GENERAL = "general"


class FeedbackProcessor:
    """
    Processes user feedback to refine generated systems.

    Capabilities:
    - Classify feedback type
    - Extract modification instructions
    - Apply changes to specifications
    - Track feedback history for learning
    """

    def __init__(self, kernel: AIONKernel):
        self.kernel = kernel

    async def process(
        self,
        feedback: str,
        session: ProgrammingSession,
    ) -> Dict[str, Any]:
        """
        Process user feedback and determine modifications.

        Args:
            feedback: User's feedback text
            session: Current programming session

        Returns:
            Modification instructions for the specification
        """
        # Classify feedback
        feedback_type = await self._classify_feedback(feedback, session)

        # Extract modifications
        modifications = await self._extract_modifications(
            feedback, feedback_type, session
        )

        logger.info(
            "Feedback processed",
            type=feedback_type,
            modifications=len(modifications.get("changes", [])),
        )

        return {
            "type": feedback_type,
            "modifications": modifications,
            "requires_regeneration": feedback_type != FeedbackType.CHANGE_NAME,
        }

    async def apply_to_spec(
        self,
        spec: Specification,
        modifications: Dict[str, Any],
    ) -> Specification:
        """Apply modifications to a specification."""
        changes = modifications.get("changes", [])

        for change in changes:
            change_type = change.get("type", "")

            if change_type == "set_name" and hasattr(spec, "name"):
                spec.name = change.get("value", spec.name)
            elif change_type == "set_description" and hasattr(spec, "description"):
                spec.description = change.get("value", spec.description)
            elif change_type == "add_parameter" and hasattr(spec, "parameters"):
                from aion.nlp.types import ParameterSpec
                spec.parameters.append(ParameterSpec(
                    name=change.get("name", "param"),
                    type=change.get("param_type", "string"),
                    description=change.get("description", ""),
                    required=change.get("required", True),
                ))
            elif change_type == "remove_parameter" and hasattr(spec, "parameters"):
                spec.parameters = [
                    p for p in spec.parameters
                    if p.name != change.get("name")
                ]

        return spec

    async def _classify_feedback(
        self,
        feedback: str,
        session: ProgrammingSession,
    ) -> str:
        """Classify the type of feedback."""
        feedback_lower = feedback.lower()

        # Quick pattern matching
        if any(w in feedback_lower for w in ["add", "include", "also need"]):
            return FeedbackType.ADD_FEATURE
        if any(w in feedback_lower for w in ["remove", "don't need", "get rid of"]):
            return FeedbackType.REMOVE_FEATURE
        if any(w in feedback_lower for w in ["change", "modify", "instead", "different"]):
            return FeedbackType.CHANGE_BEHAVIOR
        if any(w in feedback_lower for w in ["fix", "bug", "error", "wrong", "broken"]):
            return FeedbackType.FIX_BUG
        if any(w in feedback_lower for w in ["rename", "call it", "name it"]):
            return FeedbackType.CHANGE_NAME
        if any(w in feedback_lower for w in ["parameter", "argument", "input", "accept"]):
            return FeedbackType.ADD_PARAMETER
        if any(w in feedback_lower for w in ["trigger", "schedule", "when"]):
            return FeedbackType.CHANGE_TRIGGER

        return FeedbackType.GENERAL

    async def _extract_modifications(
        self,
        feedback: str,
        feedback_type: str,
        session: ProgrammingSession,
    ) -> Dict[str, Any]:
        """Extract structured modifications from feedback."""
        current_spec = session.current_spec
        spec_desc = ""
        if current_spec and hasattr(current_spec, "to_dict"):
            spec_desc = str(current_spec.to_dict())

        prompt = f"""Given this user feedback about a generated system:

Feedback: "{feedback}"
Feedback type: {feedback_type}
Current specification: {spec_desc[:500]}

Extract the specific changes needed.
Respond with ONLY valid JSON:
{{
    "changes": [
        {{
            "type": "<change_type>",
            "field": "<field_to_change>",
            "value": "<new_value>",
            "description": "<what to change>"
        }}
    ],
    "summary": "<brief summary of changes>"
}}"""

        try:
            response = await self.kernel.llm.complete(
                [{"role": "user", "content": prompt}]
            )
            content = response.content if hasattr(response, "content") else str(response)
            from aion.nlp.utils import parse_json_safe
            result = parse_json_safe(content)
            return result if result.get("changes") is not None else {"changes": [], "summary": feedback}
        except Exception as e:
            logger.debug("LLM feedback extraction failed", error=str(e))
            return {"changes": [], "summary": feedback}
