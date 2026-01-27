"""
AION Intent Parser - SOTA ensemble intent classification.

Combines three classification strategies for robust understanding:
1. Pattern-based fast matching (rule engine)
2. LLM-powered deep semantic analysis
3. Weighted ensemble scoring with calibrated confidence

Follows the same ensemble approach used in aion.conversation.routing.intent
but specialized for the NLP programming domain.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import structlog

from aion.nlp.types import (
    Complexity,
    Entity,
    EntityType,
    Intent,
    IntentType,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel
    from aion.nlp.config import NLProgrammingConfig

logger = structlog.get_logger(__name__)


# =============================================================================
# Intent Pattern Registry
# =============================================================================

# Scored patterns: (regex, weight) - higher weight = stronger signal
_INTENT_PATTERNS: Dict[IntentType, List[Tuple[str, float]]] = {
    IntentType.CREATE_TOOL: [
        (r"\bcreate\s+(?:a\s+)?(?:new\s+)?tool\b", 1.0),
        (r"\bmake\s+(?:a\s+)?(?:new\s+)?tool\b", 1.0),
        (r"\bbuild\s+(?:a\s+)?(?:new\s+)?tool\b", 1.0),
        (r"\b(?:i\s+)?need\s+(?:a\s+)?tool\b", 0.9),
        (r"\badd\s+(?:a\s+)?(?:new\s+)?tool\b", 0.9),
        (r"\btool\s+(?:that|which|to)\b", 0.8),
        (r"\bgenerate\s+(?:a\s+)?tool\b", 0.9),
        (r"\bwrite\s+(?:a\s+)?tool\b", 0.9),
    ],
    IntentType.CREATE_WORKFLOW: [
        (r"\bcreate\s+(?:a\s+)?(?:new\s+)?workflow\b", 1.0),
        (r"\bautomation?\s+(?:that|which|for|to)\b", 0.9),
        (r"\bwhen\s+.+\s+then\s+", 0.85),
        (r"\bevery\s+(?:day|hour|minute|week|morning|evening|night)\b", 0.9),
        (r"\bschedule\s+(?:a\s+)?(?:task|job|process)\b", 0.85),
        (r"\bif\s+.+\s+(?:then\s+)?(?:do|run|execute|trigger)\b", 0.8),
        (r"\bpipeline\s+(?:that|which|to|for)\b", 0.8),
        (r"\bautomatically\s+", 0.7),
        (r"\bworkflow\s+(?:that|which|to)\b", 0.85),
    ],
    IntentType.CREATE_AGENT: [
        (r"\bcreate\s+(?:a\s+)?(?:an?\s+)?(?:new\s+)?agent\b", 1.0),
        (r"\bbuild\s+(?:a\s+)?(?:an?\s+)?(?:new\s+)?agent\b", 1.0),
        (r"\b(?:i\s+)?need\s+(?:a\s+)?(?:an?\s+)?agent\b", 0.9),
        (r"\bassistant\s+(?:that|who|which)\b", 0.85),
        (r"\bbot\s+(?:that|who|which)\b", 0.85),
        (r"\bai\s+(?:that|who|which)\s+(?:can|will|should)\b", 0.8),
        (r"\bautonomous\s+(?:system|process|worker)\b", 0.8),
        (r"\bmonitor(?:s|ing)?\s+(?:and|then|to)\b", 0.7),
    ],
    IntentType.CREATE_API: [
        (r"\bcreate\s+(?:a\s+)?(?:an?\s+)?(?:new\s+)?(?:rest\s*)?api\b", 1.0),
        (r"\bbuild\s+(?:a\s+)?(?:an?\s+)?(?:rest\s*)?api\b", 1.0),
        (r"\b(?:rest|graphql)\s*api\b", 0.9),
        (r"\bendpoint(?:s)?\s+(?:for|that|which)\b", 0.85),
        (r"\bweb\s*service\s+(?:for|that|which)\b", 0.85),
        (r"\bcrud\s+(?:api|endpoint|operation)\b", 0.9),
    ],
    IntentType.CREATE_INTEGRATION: [
        (r"\bconnect\s+\w+\s+(?:with|to)\b", 0.9),
        (r"\bintegrate\s+\w+\s+(?:with|to|into)\b", 1.0),
        (r"\bsync\s+.+\s+(?:with|to|from)\b", 0.9),
        (r"\blink\s+.+\s+(?:with|to)\b", 0.85),
        (r"\bbridge\s+(?:between)?\b", 0.8),
        (r"\bpull\s+(?:data\s+)?from\s+\w+\s+(?:and\s+)?(?:push|send|write)\s+to\b", 0.9),
    ],
    IntentType.MODIFY_EXISTING: [
        (r"\bmodify\s+(?:the\s+)?", 0.9),
        (r"\bchange\s+(?:the\s+)?", 0.85),
        (r"\bupdate\s+(?:the\s+)?", 0.85),
        (r"\bedit\s+(?:the\s+)?", 0.85),
        (r"\bfix\s+(?:the\s+)?", 0.9),
        (r"\bimprove\s+(?:the\s+)?", 0.8),
        (r"\brefactor\s+", 0.8),
        (r"\boptimize\s+", 0.8),
        (r"\badd\s+.+\s+to\s+(?:the\s+)?(?:existing|current)\b", 0.9),
    ],
    IntentType.DELETE: [
        (r"\bdelete\s+(?:the\s+)?", 0.95),
        (r"\bremove\s+(?:the\s+)?", 0.9),
        (r"\bdestroy\s+(?:the\s+)?", 0.95),
        (r"\buninstall\s+", 0.9),
        (r"\bdisable\s+(?:the\s+)?", 0.8),
    ],
    IntentType.DEBUG: [
        (r"\bdebug\s+", 0.95),
        (r"\bwhat(?:'s|\s+is)\s+wrong\s+with\b", 0.9),
        (r"\bwhy\s+(?:is|does|isn't|doesn't|won't|can't)\b", 0.85),
        (r"\btroubleshoot\s+", 0.9),
        (r"\bnot\s+working\b", 0.8),
        (r"\berror\s+(?:in|with|from)\b", 0.85),
    ],
    IntentType.TEST: [
        (r"\btest\s+(?:the\s+)?", 0.9),
        (r"\btry\s+(?:running|executing)\b", 0.85),
        (r"\brun\s+(?:the\s+)?", 0.8),
        (r"\bexecute\s+(?:the\s+)?", 0.8),
        (r"\bverify\s+(?:that\s+)?", 0.85),
        (r"\bvalidate\s+", 0.85),
    ],
    IntentType.EXPLAIN: [
        (r"\bexplain\s+", 0.95),
        (r"\bhow\s+does\s+", 0.9),
        (r"\bwhat\s+(?:does|is)\s+", 0.8),
        (r"\bdescribe\s+", 0.85),
        (r"\bshow\s+me\s+(?:how|what)\b", 0.85),
    ],
    IntentType.LIST: [
        (r"\blist\s+(?:all\s+)?(?:my\s+)?", 0.9),
        (r"\bshow\s+(?:all\s+)?(?:my\s+)?", 0.85),
        (r"\bwhat\s+(?:tools|workflows|agents|apis)\s+", 0.85),
    ],
    IntentType.STATUS: [
        (r"\bstatus\s+(?:of\s+)?", 0.9),
        (r"\bhow\s+is\s+.+\s+(?:doing|performing|running)\b", 0.85),
        (r"\bhealth\s+(?:of|check)\b", 0.85),
        (r"\bmetrics\s+(?:for|of)\b", 0.8),
    ],
    IntentType.DEPLOY: [
        (r"\bdeploy\s+", 0.95),
        (r"\bactivate\s+", 0.85),
        (r"\blaunch\s+", 0.85),
        (r"\bpublish\s+", 0.85),
        (r"\bgo\s+live\b", 0.9),
    ],
    IntentType.ROLLBACK: [
        (r"\brollback\s+", 0.95),
        (r"\brevert\s+", 0.9),
        (r"\bundo\s+", 0.85),
        (r"\brestore\s+(?:previous|last|old)\b", 0.9),
    ],
}


# Complexity indicators
_COMPLEXITY_SIGNALS: Dict[Complexity, List[str]] = {
    Complexity.TRIVIAL: [
        r"\bsimple\b", r"\bbasic\b", r"\bjust\b", r"\bonly\b", r"\bquick\b",
    ],
    Complexity.COMPLEX: [
        r"\bcomplex\b", r"\badvanced\b", r"\bsophisticated\b",
        r"\bmultiple\s+(?:steps|stages|phases)\b",
        r"\bwith\s+(?:error\s+handling|retries|caching|auth)\b",
    ],
    Complexity.EXPERT: [
        r"\bscalable\b", r"\bdistributed\b", r"\bhigh\s*(?:availability|performance)\b",
        r"\bmicroservice\b", r"\bevent\s*(?:driven|sourcing)\b",
    ],
}


class IntentParser:
    """
    SOTA ensemble intent classifier for NLP programming requests.

    Combines pattern matching (fast, deterministic) with LLM analysis
    (deep, semantic) using calibrated confidence weighting.
    """

    def __init__(self, kernel: AIONKernel, config: Optional[NLProgrammingConfig] = None):
        self.kernel = kernel
        self._config = config

    @property
    def _weights(self) -> Tuple[float, float]:
        if self._config:
            return (self._config.intent.pattern_weight, self._config.intent.llm_weight)
        return (0.3, 0.5)

    async def parse(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Intent:
        """
        Parse user input into a structured Intent.

        Pipeline:
        1. Normalize input
        2. Pattern-based classification (fast path)
        3. LLM-based deep analysis (if ambiguous or for enrichment)
        4. Entity extraction
        5. Complexity estimation
        6. Clarification check
        7. Ensemble scoring
        """
        normalized = self._normalize(user_input)

        # Phase 1: Pattern-based classification
        pattern_scores = self._pattern_classify(normalized)
        pattern_type, pattern_conf = self._best_match(pattern_scores)

        # Phase 2: LLM deep analysis
        llm_result = await self._llm_classify(user_input, context)
        llm_type = llm_result.get("intent_type", pattern_type.value)
        llm_conf = llm_result.get("confidence", 0.0)

        # Phase 3: Ensemble
        final_type, final_conf = self._ensemble_score(
            pattern_type, pattern_conf,
            llm_type, llm_conf,
        )

        # Phase 4: Extract entities
        entities = self._extract_basic_entities(user_input, normalized)

        # Merge LLM-extracted entities
        if llm_result.get("entities"):
            for ent_data in llm_result["entities"]:
                ent_type = self._map_entity_type(ent_data.get("type", ""))
                if ent_type:
                    entities.append(Entity(
                        type=ent_type,
                        value=ent_data.get("value", ""),
                        confidence=ent_data.get("confidence", 0.7),
                        metadata=tuple(ent_data.get("metadata", {}).items()),
                    ))

        # Phase 5: Complexity estimation
        complexity = self._estimate_complexity(normalized, entities)

        # Build intent
        intent = Intent(
            type=final_type,
            confidence=final_conf,
            complexity=complexity,
            entities=entities,
            raw_input=user_input,
            normalized_input=normalized,
            parameters=llm_result.get("parameters", {}),
            predictions=[
                {"source": "pattern", "type": pattern_type.value, "confidence": pattern_conf},
                {"source": "llm", "type": llm_type, "confidence": llm_conf},
            ],
        )

        # Phase 6: Clarification check
        intent = self._check_clarification_needed(intent)

        # Phase 7: Multi-intent detection
        intent = self._detect_multi_intent(intent, normalized)

        logger.info(
            "Intent parsed",
            type=intent.type.value,
            confidence=round(intent.confidence, 3),
            complexity=intent.complexity.value,
            entities=len(intent.entities),
            needs_clarification=intent.needs_clarification,
        )

        return intent

    # =========================================================================
    # Normalization
    # =========================================================================

    def _normalize(self, text: str) -> str:
        """Normalize input for pattern matching."""
        text = text.lower().strip()
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove leading filler words
        text = re.sub(r"^(?:please|can you|could you|i(?:'d)? (?:want|like) (?:you )?to)\s+", "", text)
        return text

    # =========================================================================
    # Pattern Classification
    # =========================================================================

    def _pattern_classify(self, text: str) -> Dict[IntentType, float]:
        """Score each intent type using weighted pattern matching."""
        scores: Dict[IntentType, float] = {}

        for intent_type, patterns in _INTENT_PATTERNS.items():
            type_score = 0.0
            match_count = 0
            for pattern, weight in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    type_score += weight
                    match_count += 1

            if match_count > 0:
                # Average match score, boosted by number of matching patterns
                avg_score = type_score / match_count
                multi_match_bonus = min(0.15, 0.05 * (match_count - 1))
                scores[intent_type] = min(1.0, avg_score + multi_match_bonus)

        return scores

    def _best_match(self, scores: Dict[IntentType, float]) -> Tuple[IntentType, float]:
        """Get the best matching intent type."""
        if not scores:
            return IntentType.CREATE_TOOL, 0.2

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best_type, scores[best_type]

    # =========================================================================
    # LLM Classification
    # =========================================================================

    async def _llm_classify(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use LLM for deep semantic analysis."""
        intent_types_str = ", ".join(t.value for t in IntentType)
        entity_types_str = ", ".join(t.value for t in EntityType)

        prompt = f"""Analyze this natural language programming request and extract structured information.

Request: "{user_input}"

{f'Context: {json.dumps(context)}' if context else ''}

Available intent types: {intent_types_str}
Available entity types: {entity_types_str}

Respond with ONLY valid JSON (no markdown):
{{
    "intent_type": "<intent_type>",
    "confidence": <0.0-1.0>,
    "parameters": {{
        "name": "<system name or null>",
        "description": "<what it should do>",
        "inputs": [
            {{"name": "<param>", "type": "<type>", "description": "<desc>", "required": true}}
        ],
        "outputs": [
            {{"name": "<output>", "type": "<type>"}}
        ],
        "triggers": ["<trigger descriptions>"],
        "constraints": ["<constraints>"],
        "notes": "<implementation notes>"
    }},
    "entities": [
        {{"type": "<entity_type>", "value": "<value>", "confidence": <0.0-1.0>}}
    ],
    "clarification_needed": ["<question if unclear>"]
}}"""

        try:
            response = await self.kernel.llm.complete(
                [{"role": "user", "content": prompt}]
            )
            content = response.content if hasattr(response, "content") else str(response)
            return self._parse_json_safe(content)
        except Exception as e:
            logger.warning("LLM classification failed", error=str(e))
            return {}

    # =========================================================================
    # Ensemble Scoring
    # =========================================================================

    def _ensemble_score(
        self,
        pattern_type: IntentType,
        pattern_conf: float,
        llm_type_str: str,
        llm_conf: float,
    ) -> Tuple[IntentType, float]:
        """Combine pattern and LLM scores using calibrated weighting."""
        pattern_w, llm_w = self._weights

        # Parse LLM type
        try:
            llm_type = IntentType(llm_type_str)
        except ValueError:
            llm_type = pattern_type
            llm_conf *= 0.5  # Penalize invalid type

        # If both agree, boost confidence
        if pattern_type == llm_type:
            combined_conf = min(1.0, (pattern_conf * pattern_w + llm_conf * llm_w) / (pattern_w + llm_w) + 0.1)
            return pattern_type, combined_conf

        # Disagreement: use weighted voting
        pattern_score = pattern_conf * pattern_w
        llm_score = llm_conf * llm_w

        if llm_score > pattern_score:
            # LLM wins but with reduced confidence due to disagreement
            return llm_type, llm_score / (pattern_w + llm_w) * 0.9
        else:
            return pattern_type, pattern_score / (pattern_w + llm_w) * 0.9

    # =========================================================================
    # Entity Extraction (basic - EntityExtractor has the full pipeline)
    # =========================================================================

    def _extract_basic_entities(self, original: str, normalized: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities: List[Entity] = []

        # Quoted strings -> potential names
        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`', original):
            value = match.group(1) or match.group(2) or match.group(3)
            entities.append(Entity(
                type=EntityType.TOOL_NAME,
                value=value,
                confidence=0.9,
                span_start=match.start(),
                span_end=match.end(),
                metadata=(("source", "quoted"),),
            ))

        # Schedule patterns
        schedule_patterns = [
            (r"\bevery\s+(day|hour|minute|week|month)\b", "interval"),
            (r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", "time"),
            (r"\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", "day"),
            (r"\b(\d+)\s*(?:times?\s+per|/)\s*(day|hour|week|month)\b", "frequency"),
            (r"\bdaily\b", "interval"),
            (r"\bhourly\b", "interval"),
            (r"\bweekly\b", "interval"),
        ]
        for pattern, subtype in schedule_patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                entities.append(Entity(
                    type=EntityType.SCHEDULE,
                    value=match.group(0),
                    confidence=0.85,
                    span_start=match.start(),
                    span_end=match.end(),
                    metadata=(("subtype", subtype),),
                ))

        # Trigger patterns
        trigger_patterns = [
            (r"\bwhen\s+(.+?)\s+(?:then|,)", "condition"),
            (r"\bif\s+(.+?)\s+(?:then|,|do\b)", "condition"),
            (r"\bon\s+(file\s+change|new\s+(?:email|message|event)|webhook|push|commit)\b", "event"),
        ]
        for pattern, subtype in trigger_patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                value = match.group(1) if match.lastindex else match.group(0)
                entities.append(Entity(
                    type=EntityType.TRIGGER,
                    value=value,
                    confidence=0.8,
                    span_start=match.start(),
                    span_end=match.end(),
                    metadata=(("subtype", subtype),),
                ))

        # Data source/target patterns
        for match in re.finditer(r"\bfrom\s+(\w+(?:\s+\w+)?)\b", normalized):
            entities.append(Entity(
                type=EntityType.DATA_SOURCE,
                value=match.group(1),
                confidence=0.75,
                span_start=match.start(),
                span_end=match.end(),
            ))

        for match in re.finditer(r"\b(?:to|into)\s+(\w+(?:\s+\w+)?)\b", normalized):
            entities.append(Entity(
                type=EntityType.DATA_TARGET,
                value=match.group(1),
                confidence=0.75,
                span_start=match.start(),
                span_end=match.end(),
            ))

        # API endpoint / URL
        for match in re.finditer(r"https?://[^\s]+", original):
            entities.append(Entity(
                type=EntityType.API_ENDPOINT,
                value=match.group(0),
                confidence=0.95,
                span_start=match.start(),
                span_end=match.end(),
            ))

        return entities

    # =========================================================================
    # Complexity Estimation
    # =========================================================================

    def _estimate_complexity(self, text: str, entities: List[Entity]) -> Complexity:
        """Estimate task complexity from signals in text and entities."""
        scores: Dict[Complexity, float] = {c: 0.0 for c in Complexity}

        for complexity, patterns in _COMPLEXITY_SIGNALS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[complexity] += 1.0

        # Entity-based complexity: more entities = more complex
        entity_count = len(entities)
        if entity_count <= 2:
            scores[Complexity.SIMPLE] += 1.0
        elif entity_count <= 5:
            scores[Complexity.MODERATE] += 1.0
        else:
            scores[Complexity.COMPLEX] += 1.0

        # Word count heuristic
        word_count = len(text.split())
        if word_count < 10:
            scores[Complexity.TRIVIAL] += 0.5
        elif word_count > 50:
            scores[Complexity.COMPLEX] += 0.5

        # Default to SIMPLE if no strong signals
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0.0:
            return Complexity.SIMPLE
        return best

    # =========================================================================
    # Clarification
    # =========================================================================

    def _check_clarification_needed(self, intent: Intent) -> Intent:
        """Check if the intent needs user clarification."""
        questions: List[str] = []

        # Low confidence -> ask for clarification
        threshold = 0.7
        if self._config:
            threshold = self._config.intent.clarification_threshold
        if intent.confidence < threshold:
            questions.append(
                f"I'm not fully sure what you'd like to create. "
                f"Could you clarify: are you looking to create a "
                f"{intent.type.value.replace('create_', '').replace('_', ' ')}?"
            )

        # Missing critical info based on intent type
        if intent.type == IntentType.CREATE_TOOL:
            if not intent.name:
                questions.append("What would you like to name this tool?")
            if not intent.parameters.get("description") and not intent.description:
                questions.append("What should this tool do? Please describe its purpose.")

        elif intent.type == IntentType.CREATE_WORKFLOW:
            if not intent.get_entities(EntityType.TRIGGER) and not intent.parameters.get("triggers"):
                questions.append("What should trigger this workflow? (e.g., schedule, event, manual)")

        elif intent.type == IntentType.CREATE_AGENT:
            if not intent.name:
                questions.append("What would you like to name this agent?")
            if not intent.parameters.get("description") and not intent.description:
                questions.append("What is the primary goal of this agent?")

        elif intent.type == IntentType.CREATE_API:
            if not intent.parameters.get("description") and not intent.description:
                questions.append("What resources should this API manage?")

        elif intent.type == IntentType.CREATE_INTEGRATION:
            if not intent.get_entities(EntityType.DATA_SOURCE):
                questions.append("What system should this integration connect from?")
            if not intent.get_entities(EntityType.DATA_TARGET):
                questions.append("What system should this integration connect to?")

        if questions:
            intent.needs_clarification = True
            intent.clarification_questions.extend(questions)
            intent.ambiguity_score = 1.0 - intent.confidence

        return intent

    # =========================================================================
    # Multi-Intent Detection
    # =========================================================================

    def _detect_multi_intent(self, intent: Intent, text: str) -> Intent:
        """Detect compound requests with multiple intents."""
        # Look for conjunctions that indicate multiple requests
        multi_patterns = [
            r"\b(?:and\s+also|and\s+then|also|additionally)\s+(?:create|build|make)\b",
            r"\bcreate\s+(?:a\s+)?(?:tool|workflow|agent).+\band\s+(?:a\s+)?(?:tool|workflow|agent)\b",
        ]

        for pattern in multi_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                intent.parameters["multi_intent_detected"] = True
                # Don't split yet - let the engine handle it
                break

        return intent

    # =========================================================================
    # Utilities
    # =========================================================================

    def _map_entity_type(self, type_str: str) -> Optional[EntityType]:
        """Map string to EntityType, returning None if invalid."""
        try:
            return EntityType(type_str)
        except ValueError:
            # Try common aliases
            aliases = {
                "name": EntityType.TOOL_NAME,
                "source": EntityType.DATA_SOURCE,
                "target": EntityType.DATA_TARGET,
                "trigger": EntityType.TRIGGER,
                "schedule": EntityType.SCHEDULE,
                "endpoint": EntityType.API_ENDPOINT,
                "param": EntityType.PARAMETER,
                "format": EntityType.DATA_FORMAT,
            }
            return aliases.get(type_str.lower())

    def _parse_json_safe(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        return {}
