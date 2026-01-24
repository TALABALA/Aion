"""
AION Entity Extractor

Extract entities and relationships from text using LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.systems.knowledge.types import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    Triple,
    Provenance,
)

logger = structlog.get_logger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""
    min_confidence: float = 0.5
    extract_relationships: bool = True
    extract_properties: bool = True
    max_entities: int = 50
    max_relationships: int = 100
    deduplicate: bool = True


@dataclass
class ExtractionResult:
    """Result from entity extraction."""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    raw_response: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "triples": [t.to_string() for t in self.triples],
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
        }


class EntityExtractor:
    """
    Extract entities and relationships from text using LLM.

    Features:
    - Named entity recognition
    - Relationship extraction
    - Property extraction
    - Coreference resolution
    - Confidence scoring
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._llm = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the extractor."""
        if self._initialized:
            return

        try:
            from aion.core.llm import LLMAdapter
            self._llm = LLMAdapter()
            await self._llm.initialize()
            self._initialized = True
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}")

    async def extract(
        self,
        text: str,
        context: Optional[str] = None,
        existing_entities: Optional[List[Entity]] = None,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from text.

        Args:
            text: Text to extract from
            context: Optional context about the text
            existing_entities: Known entities for linking

        Returns:
            ExtractionResult with entities and relationships
        """
        if not self._initialized:
            await self.initialize()

        if not self._llm:
            logger.warning("LLM not available, using fallback extraction")
            return self._fallback_extract(text)

        # Build prompt
        prompt = self._build_extraction_prompt(text, context, existing_entities)

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=self._get_system_prompt(),
            )

            raw_text = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            result = self._parse_extraction_response(raw_text, text)
            result.raw_response = raw_text

            # Link to existing entities
            if existing_entities and self.config.deduplicate:
                result = self._link_entities(result, existing_entities)

            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return self._fallback_extract(text)

    def _get_system_prompt(self) -> str:
        """Get system prompt for extraction."""
        return """You are an expert knowledge extraction system. Extract structured information from text.

Your task is to identify:
1. ENTITIES: Named things (people, organizations, projects, concepts, etc.)
2. RELATIONSHIPS: How entities are connected
3. PROPERTIES: Attributes and facts about entities

Be precise and extract only what is explicitly stated or strongly implied.
Assign confidence scores (0.0-1.0) based on how certain the extraction is."""

    def _build_extraction_prompt(
        self,
        text: str,
        context: Optional[str],
        existing_entities: Optional[List[Entity]],
    ) -> str:
        """Build the extraction prompt."""
        prompt = f"""Extract entities and relationships from this text.

TEXT:
{text}

"""

        if context:
            prompt += f"""CONTEXT:
{context}

"""

        if existing_entities:
            entity_list = ", ".join(e.name for e in existing_entities[:20])
            prompt += f"""KNOWN ENTITIES (try to link to these):
{entity_list}

"""

        prompt += """Respond with ONLY a valid JSON object:
{
    "entities": [
        {
            "name": "Entity Name",
            "type": "person|organization|project|concept|event|location|document|tool",
            "description": "Brief description",
            "properties": {"key": "value"},
            "confidence": 0.9,
            "aliases": ["alternate name"]
        }
    ],
    "relationships": [
        {
            "source": "Source Entity Name",
            "target": "Target Entity Name",
            "type": "works_for|manages|created|part_of|related_to|knows|caused|located_in|depends_on|member_of",
            "properties": {},
            "confidence": 0.8
        }
    ]
}

Entity types: person, organization, project, concept, event, location, document, tool, task, goal
Relationship types: works_for, manages, reports_to, member_of, knows, collaborates_with, created, owns, authored, is_a, part_of, related_to, similar_to, caused, located_in, depends_on, blocks, assigned_to

Extract ALL entities and relationships mentioned. Be thorough but accurate."""

        return prompt

    def _parse_extraction_response(
        self,
        response: str,
        source_text: str,
    ) -> ExtractionResult:
        """Parse LLM response into structured result."""
        result = ExtractionResult()

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.warning("No JSON found in extraction response")
            return result

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return result

        # Parse entities
        entity_map: Dict[str, Entity] = {}

        for e_data in data.get("entities", []):
            confidence = float(e_data.get("confidence", 0.8))

            if confidence < self.config.min_confidence:
                continue

            entity = Entity(
                name=e_data.get("name", "Unknown"),
                entity_type=self._parse_entity_type(e_data.get("type", "concept")),
                description=e_data.get("description", ""),
                aliases=e_data.get("aliases", []),
                properties=e_data.get("properties", {}),
                confidence=confidence,
                provenance=Provenance(
                    source_type="extraction",
                    extraction_method="llm",
                    extraction_confidence=confidence,
                ),
            )

            result.entities.append(entity)
            entity_map[entity.name.lower()] = entity

            # Also map aliases
            for alias in entity.aliases:
                entity_map[alias.lower()] = entity

        # Parse relationships
        for r_data in data.get("relationships", []):
            confidence = float(r_data.get("confidence", 0.7))

            if confidence < self.config.min_confidence:
                continue

            source_name = r_data.get("source", "").lower()
            target_name = r_data.get("target", "").lower()

            source_entity = entity_map.get(source_name)
            target_entity = entity_map.get(target_name)

            if not source_entity or not target_entity:
                continue

            rel = Relationship(
                source_id=source_entity.id,
                target_id=target_entity.id,
                relation_type=self._parse_relation_type(r_data.get("type", "related_to")),
                properties=r_data.get("properties", {}),
                confidence=confidence,
                provenance=Provenance(
                    source_type="extraction",
                    extraction_method="llm",
                    extraction_confidence=confidence,
                ),
            )

            result.relationships.append(rel)

            # Create triple
            triple = Triple(
                subject=source_entity,
                predicate=rel.relation_type,
                object=target_entity,
                relationship=rel,
                confidence=confidence,
            )
            result.triples.append(triple)

        # Truncate if needed
        if len(result.entities) > self.config.max_entities:
            result.entities = sorted(
                result.entities,
                key=lambda e: e.confidence,
                reverse=True,
            )[:self.config.max_entities]

        if len(result.relationships) > self.config.max_relationships:
            result.relationships = sorted(
                result.relationships,
                key=lambda r: r.confidence,
                reverse=True,
            )[:self.config.max_relationships]

        return result

    def _fallback_extract(self, text: str) -> ExtractionResult:
        """Simple regex-based fallback extraction."""
        result = ExtractionResult()

        # Extract potential entity names (capitalized words/phrases)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, text)

        seen = set()
        for match in matches:
            if match.lower() not in seen and len(match) > 2:
                seen.add(match.lower())
                entity = Entity(
                    name=match,
                    entity_type=EntityType.CONCEPT,
                    confidence=0.5,
                    provenance=Provenance(
                        source_type="extraction",
                        extraction_method="regex",
                        extraction_confidence=0.5,
                    ),
                )
                result.entities.append(entity)

        return result

    def _link_entities(
        self,
        result: ExtractionResult,
        existing: List[Entity],
    ) -> ExtractionResult:
        """Link extracted entities to existing ones."""
        from aion.systems.knowledge.extraction.linker import EntityLinker
        linker = EntityLinker()

        linked_entities = []
        id_map: Dict[str, str] = {}  # old_id -> linked_id

        for entity in result.entities:
            match = linker.find_best_match(entity, existing)
            if match:
                # Use existing entity ID
                id_map[entity.id] = match.id
                # Merge properties
                match.properties.update(entity.properties)
                linked_entities.append(match)
            else:
                linked_entities.append(entity)

        # Update relationship IDs
        for rel in result.relationships:
            if rel.source_id in id_map:
                rel.source_id = id_map[rel.source_id]
            if rel.target_id in id_map:
                rel.target_id = id_map[rel.target_id]

        result.entities = linked_entities

        return result

    def _parse_entity_type(self, type_str: str) -> EntityType:
        """Parse entity type string."""
        mapping = {
            "person": EntityType.PERSON,
            "human": EntityType.PERSON,
            "organization": EntityType.ORGANIZATION,
            "org": EntityType.ORGANIZATION,
            "company": EntityType.ORGANIZATION,
            "team": EntityType.TEAM,
            "project": EntityType.PROJECT,
            "concept": EntityType.CONCEPT,
            "idea": EntityType.CONCEPT,
            "topic": EntityType.TOPIC,
            "event": EntityType.EVENT,
            "location": EntityType.LOCATION,
            "place": EntityType.LOCATION,
            "document": EntityType.DOCUMENT,
            "tool": EntityType.TOOL,
            "task": EntityType.TASK,
            "goal": EntityType.GOAL,
        }
        return mapping.get(type_str.lower(), EntityType.CONCEPT)

    def _parse_relation_type(self, type_str: str) -> RelationType:
        """Parse relation type string."""
        mapping = {
            "works_for": RelationType.WORKS_FOR,
            "works for": RelationType.WORKS_FOR,
            "employed_by": RelationType.WORKS_FOR,
            "manages": RelationType.MANAGES,
            "reports_to": RelationType.REPORTS_TO,
            "reports to": RelationType.REPORTS_TO,
            "member_of": RelationType.MEMBER_OF,
            "member of": RelationType.MEMBER_OF,
            "belongs_to": RelationType.MEMBER_OF,
            "knows": RelationType.KNOWS,
            "collaborates_with": RelationType.COLLABORATES_WITH,
            "collaborates with": RelationType.COLLABORATES_WITH,
            "created": RelationType.CREATED,
            "made": RelationType.CREATED,
            "built": RelationType.CREATED,
            "owns": RelationType.OWNS,
            "authored": RelationType.AUTHORED,
            "wrote": RelationType.AUTHORED,
            "is_a": RelationType.IS_A,
            "is a": RelationType.IS_A,
            "type_of": RelationType.IS_A,
            "part_of": RelationType.PART_OF,
            "part of": RelationType.PART_OF,
            "component_of": RelationType.PART_OF,
            "related_to": RelationType.RELATED_TO,
            "related to": RelationType.RELATED_TO,
            "similar_to": RelationType.SIMILAR_TO,
            "similar to": RelationType.SIMILAR_TO,
            "caused": RelationType.CAUSED,
            "causes": RelationType.CAUSED,
            "led_to": RelationType.CAUSED,
            "located_in": RelationType.LOCATED_IN,
            "located in": RelationType.LOCATED_IN,
            "based_in": RelationType.LOCATED_IN,
            "depends_on": RelationType.DEPENDS_ON,
            "depends on": RelationType.DEPENDS_ON,
            "requires": RelationType.DEPENDS_ON,
            "blocks": RelationType.BLOCKS,
            "assigned_to": RelationType.ASSIGNED_TO,
            "assigned to": RelationType.ASSIGNED_TO,
        }
        return mapping.get(type_str.lower(), RelationType.RELATED_TO)

    async def extract_triples(self, text: str) -> List[Triple]:
        """Extract subject-predicate-object triples from text."""
        result = await self.extract(text)
        return result.triples
