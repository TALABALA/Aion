"""
AION Entity Extractor - Deep entity extraction from natural language.

Extracts structured entities including parameters, data flows,
schedules, conditions, API details, and more.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import Entity, EntityType, Intent

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel
    from aion.nlp.config import NLProgrammingConfig

logger = structlog.get_logger(__name__)


# Type inference patterns
_TYPE_PATTERNS: Dict[str, str] = {
    r"\b(?:text|string|name|title|label|message|description)\b": "string",
    r"\b(?:number|count|amount|quantity|total|size|length)\b": "int",
    r"\b(?:price|cost|rate|percentage|ratio|score|weight)\b": "float",
    r"\b(?:flag|enabled|disabled|active|toggle|boolean)\b": "bool",
    r"\b(?:date|time|datetime|timestamp|when|deadline)\b": "datetime",
    r"\b(?:list|array|items|collection|set)\b": "list",
    r"\b(?:map|dict|object|record|config|settings)\b": "dict",
    r"\b(?:url|link|endpoint|uri|href)\b": "url",
    r"\b(?:email|address)\b": "email",
    r"\b(?:file|path|directory|folder)\b": "path",
    r"\b(?:json|xml|csv|yaml)\b": "format",
}

# Service name patterns for integration detection
_KNOWN_SERVICES = {
    "slack", "discord", "teams", "telegram", "whatsapp",
    "gmail", "outlook", "email", "sendgrid", "mailchimp",
    "github", "gitlab", "bitbucket", "jira", "trello", "asana",
    "aws", "gcp", "azure", "docker", "kubernetes",
    "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "stripe", "paypal", "shopify",
    "salesforce", "hubspot", "zendesk",
    "twilio", "firebase", "supabase",
    "notion", "airtable", "google sheets",
    "openai", "anthropic", "huggingface",
    "s3", "gcs", "dropbox", "drive",
}


class EntityExtractor:
    """
    Deep entity extractor for NLP programming requests.

    Extracts:
    - Named entities (tool names, service names)
    - Parameters with type inference
    - Data flow (sources, targets, formats)
    - Temporal patterns (schedules, timeouts)
    - Constraints and requirements
    - API details (endpoints, methods, auth)
    """

    def __init__(self, kernel: AIONKernel, config: Optional[NLProgrammingConfig] = None):
        self.kernel = kernel
        self._config = config

    async def extract(self, intent: Intent) -> Intent:
        """
        Enrich an intent with deeply extracted entities.

        Args:
            intent: Intent with basic entities already extracted

        Returns:
            Enriched intent with additional entities
        """
        text = intent.raw_input
        normalized = intent.normalized_input or text.lower()

        # Extract each entity category
        new_entities: List[Entity] = []

        new_entities.extend(self._extract_parameters(text, normalized))
        new_entities.extend(self._extract_services(text, normalized))
        new_entities.extend(self._extract_api_details(text, normalized))
        new_entities.extend(self._extract_error_handling(text, normalized))
        new_entities.extend(self._extract_constraints(text, normalized))
        new_entities.extend(self._extract_formats(text, normalized))

        # Deduplicate entities (by type + value)
        seen = {(e.type, e.value) for e in intent.entities}
        for entity in new_entities:
            key = (entity.type, entity.value)
            if key not in seen:
                intent.entities.append(entity)
                seen.add(key)

        # LLM-enhanced extraction for complex requests
        if intent.complexity.value in ("complex", "expert"):
            intent = await self._llm_extract(intent)

        logger.debug(
            "Entity extraction complete",
            total_entities=len(intent.entities),
            new_entities=len(new_entities),
        )

        return intent

    def _extract_parameters(self, text: str, normalized: str) -> List[Entity]:
        """Extract parameter definitions from text."""
        entities: List[Entity] = []

        # Pattern: "takes a <type> <name>"
        param_patterns = [
            r"\btakes?\s+(?:a\s+)?(\w+)\s+(?:called|named)\s+(\w+)",
            r"\bwith\s+(?:a\s+)?(\w+)\s+parameter\s+(?:called|named)?\s*(\w+)?",
            r"\binput(?:s)?:\s*(.+?)(?:\.|$)",
            r"\bparameter(?:s)?:\s*(.+?)(?:\.|$)",
            r"\baccepts?\s+(.+?)\s+(?:as\s+(?:input|parameter))",
            r"\b(\w+)\s*:\s*(\w+)\s*(?:=\s*(\S+))?\s*(?:,|\)|$)",
        ]

        for pattern in param_patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                value = match.group(0)
                entities.append(Entity(
                    type=EntityType.PARAMETER,
                    value=value.strip(),
                    confidence=0.75,
                    span_start=match.start(),
                    span_end=match.end(),
                    metadata=(("source", "pattern"),),
                ))

        # Infer parameter types from context
        for type_pattern, type_name in _TYPE_PATTERNS.items():
            for match in re.finditer(type_pattern, normalized, re.IGNORECASE):
                entities.append(Entity(
                    type=EntityType.PARAMETER_TYPE,
                    value=type_name,
                    confidence=0.7,
                    span_start=match.start(),
                    span_end=match.end(),
                    metadata=(("matched_word", match.group(0)),),
                ))

        return entities

    def _extract_services(self, text: str, normalized: str) -> List[Entity]:
        """Extract service/platform references."""
        entities: List[Entity] = []

        for service in _KNOWN_SERVICES:
            pattern = r"\b" + re.escape(service) + r"\b"
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                entities.append(Entity(
                    type=EntityType.SERVICE_NAME,
                    value=service,
                    confidence=0.9,
                    span_start=match.start(),
                    span_end=match.end(),
                ))

        return entities

    def _extract_api_details(self, text: str, normalized: str) -> List[Entity]:
        """Extract API-related details."""
        entities: List[Entity] = []

        # HTTP methods
        method_pattern = r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b"
        for match in re.finditer(method_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                type=EntityType.API_METHOD,
                value=match.group(1).upper(),
                confidence=0.9,
                span_start=match.start(),
                span_end=match.end(),
            ))

        # Auth types
        auth_patterns = [
            (r"\b(?:api\s*key|apikey)\b", "api_key"),
            (r"\b(?:oauth|oauth2)\b", "oauth2"),
            (r"\b(?:bearer\s+token|jwt)\b", "bearer"),
            (r"\b(?:basic\s+auth)\b", "basic"),
            (r"\b(?:hmac|signature)\b", "hmac"),
        ]
        for pattern, auth_type in auth_patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                entities.append(Entity(
                    type=EntityType.AUTH_TYPE,
                    value=auth_type,
                    confidence=0.85,
                ))

        # Rate limit
        rate_match = re.search(r"(\d+)\s*(?:requests?\s*)?(?:per|/)\s*(second|minute|hour)", normalized)
        if rate_match:
            entities.append(Entity(
                type=EntityType.RATE_LIMIT,
                value=f"{rate_match.group(1)}/{rate_match.group(2)}",
                confidence=0.85,
                span_start=rate_match.start(),
                span_end=rate_match.end(),
            ))

        return entities

    def _extract_error_handling(self, text: str, normalized: str) -> List[Entity]:
        """Extract error handling requirements."""
        entities: List[Entity] = []

        error_patterns = [
            (r"\bretry\s+(\d+)\s*times?\b", "retry"),
            (r"\b(?:on\s+)?(?:error|failure)\s*(?:,\s*)?(retry|stop|skip|alert|notify|log)\b", "on_error"),
            (r"\btimeout\s+(?:of\s+)?(\d+)\s*(seconds?|minutes?|ms)\b", "timeout"),
            (r"\bfallback\s+to\s+(.+?)(?:\.|,|$)", "fallback"),
        ]

        for pattern, subtype in error_patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                entities.append(Entity(
                    type=EntityType.ERROR_HANDLING,
                    value=match.group(0),
                    confidence=0.8,
                    span_start=match.start(),
                    span_end=match.end(),
                    metadata=(("subtype", subtype),),
                ))

        return entities

    def _extract_constraints(self, text: str, normalized: str) -> List[Entity]:
        """Extract constraints and requirements."""
        entities: List[Entity] = []

        constraint_patterns = [
            r"\bmust\s+(.+?)(?:\.|,|$)",
            r"\bshould\s+(?:not|never)\s+(.+?)(?:\.|,|$)",
            r"\bonly\s+(?:if|when)\s+(.+?)(?:\.|,|$)",
            r"\bno\s+more\s+than\s+(.+?)(?:\.|,|$)",
            r"\bat\s+(?:least|most)\s+(.+?)(?:\.|,|$)",
            r"\brequires?\s+(.+?)(?:\.|,|$)",
        ]

        for pattern in constraint_patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                entities.append(Entity(
                    type=EntityType.CONSTRAINT,
                    value=match.group(0).strip().rstrip(".,"),
                    confidence=0.75,
                    span_start=match.start(),
                    span_end=match.end(),
                ))

        return entities

    def _extract_formats(self, text: str, normalized: str) -> List[Entity]:
        """Extract data format specifications."""
        entities: List[Entity] = []

        format_patterns = [
            (r"\b(?:in|as|output)\s+(json|xml|csv|yaml|html|markdown|text|pdf)\b", "output_format"),
            (r"\b(json|xml|csv|yaml)\s+(?:format|file|data|input|output)\b", "data_format"),
        ]

        for pattern, subtype in format_patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                entities.append(Entity(
                    type=EntityType.DATA_FORMAT,
                    value=match.group(1).lower(),
                    confidence=0.85,
                    span_start=match.start(),
                    span_end=match.end(),
                    metadata=(("subtype", subtype),),
                ))

        return entities

    async def _llm_extract(self, intent: Intent) -> Intent:
        """Use LLM for deep entity extraction on complex requests."""
        prompt = f"""Extract detailed entities from this programming request:

Request: "{intent.raw_input}"

Current entities found: {[(e.type.value, e.value) for e in intent.entities]}

Please identify any additional:
1. Parameter names and their types
2. Data sources and targets
3. Constraints or validation rules
4. Error handling requirements
5. Performance requirements

Respond with ONLY valid JSON:
{{
    "additional_entities": [
        {{"type": "<entity_type>", "value": "<value>", "confidence": <0.0-1.0>}}
    ],
    "inferred_parameters": [
        {{"name": "<name>", "type": "<type>", "description": "<desc>", "required": <bool>}}
    ]
}}"""

        try:
            response = await self.kernel.llm.complete(
                [{"role": "user", "content": prompt}]
            )
            content = response.content if hasattr(response, "content") else str(response)

            from aion.nlp.utils import parse_json_safe
            data = parse_json_safe(content)

            if data.get("additional_entities"):
                for ent in data["additional_entities"]:
                    try:
                        etype = EntityType(ent["type"])
                        intent.entities.append(Entity(
                            type=etype,
                            value=ent["value"],
                            confidence=ent.get("confidence", 0.6),
                            metadata=(("source", "llm_deep"),),
                        ))
                    except (ValueError, KeyError):
                        pass

            if data.get("inferred_parameters"):
                existing = intent.parameters.get("inputs", [])
                existing.extend(data["inferred_parameters"])
                intent.parameters["inputs"] = existing

        except Exception as e:
            logger.debug("LLM entity extraction skipped", error=str(e))

        return intent
