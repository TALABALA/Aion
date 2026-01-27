"""
AION Intent Template Library - Common request templates and examples.

Provides a library of well-known intent patterns that can be used
for few-shot learning, user suggestions, and template-based generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aion.nlp.types import IntentType, SpecificationType


@dataclass(frozen=True)
class IntentTemplate:
    """A template for a common intent pattern."""

    name: str
    description: str
    intent_type: IntentType
    spec_type: SpecificationType
    example_requests: tuple[str, ...]
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    complexity: str = "simple"


# =============================================================================
# Template Library
# =============================================================================

_TEMPLATES: List[IntentTemplate] = [
    # --- Tool Templates ---
    IntentTemplate(
        name="api_fetcher",
        description="Tool that fetches data from an external API",
        intent_type=IntentType.CREATE_TOOL,
        spec_type=SpecificationType.TOOL,
        example_requests=(
            "Create a tool that fetches weather data from OpenWeather API",
            "Make a tool to get stock prices from the financial API",
            "Build a tool that queries the GitHub API for repo info",
        ),
        default_parameters={
            "inputs": [
                {"name": "endpoint", "type": "string", "required": True},
                {"name": "params", "type": "dict", "required": False},
            ],
            "return_type": "dict",
        },
        tags=("api", "http", "data"),
    ),
    IntentTemplate(
        name="data_transformer",
        description="Tool that transforms data from one format to another",
        intent_type=IntentType.CREATE_TOOL,
        spec_type=SpecificationType.TOOL,
        example_requests=(
            "Create a tool that converts CSV to JSON",
            "Make a tool to transform XML data into a Python dict",
            "Build a data normalizer tool",
        ),
        default_parameters={
            "inputs": [
                {"name": "data", "type": "any", "required": True},
                {"name": "output_format", "type": "string", "required": True},
            ],
            "return_type": "any",
        },
        tags=("transform", "data", "format"),
    ),
    IntentTemplate(
        name="text_processor",
        description="Tool for text processing and analysis",
        intent_type=IntentType.CREATE_TOOL,
        spec_type=SpecificationType.TOOL,
        example_requests=(
            "Create a tool that summarizes long text",
            "Make a sentiment analysis tool",
            "Build a tool that extracts keywords from text",
        ),
        default_parameters={
            "inputs": [{"name": "text", "type": "string", "required": True}],
            "return_type": "dict",
        },
        tags=("text", "nlp", "analysis"),
    ),
    IntentTemplate(
        name="file_processor",
        description="Tool that processes files",
        intent_type=IntentType.CREATE_TOOL,
        spec_type=SpecificationType.TOOL,
        example_requests=(
            "Create a tool that reads and parses PDF files",
            "Make a tool to process Excel spreadsheets",
            "Build a tool that scans images for text",
        ),
        default_parameters={
            "inputs": [{"name": "file_path", "type": "string", "required": True}],
            "return_type": "dict",
        },
        tags=("file", "io", "processing"),
    ),

    # --- Workflow Templates ---
    IntentTemplate(
        name="scheduled_report",
        description="Workflow that generates and sends reports on a schedule",
        intent_type=IntentType.CREATE_WORKFLOW,
        spec_type=SpecificationType.WORKFLOW,
        example_requests=(
            "Every morning at 8am, generate a report and email it",
            "Create a daily summary workflow",
            "Build a weekly metrics report pipeline",
        ),
        default_parameters={
            "trigger_type": "schedule",
            "steps": ["gather_data", "generate_report", "send_notification"],
        },
        tags=("report", "schedule", "notification"),
    ),
    IntentTemplate(
        name="event_reactor",
        description="Workflow triggered by an event",
        intent_type=IntentType.CREATE_WORKFLOW,
        spec_type=SpecificationType.WORKFLOW,
        example_requests=(
            "When a new file is uploaded, process and index it",
            "When a support ticket is created, notify the team",
            "On new GitHub push, run tests and deploy",
        ),
        default_parameters={
            "trigger_type": "event",
            "on_error": "retry",
        },
        tags=("event", "reactive", "automation"),
    ),
    IntentTemplate(
        name="data_pipeline",
        description="Workflow that processes data through stages",
        intent_type=IntentType.CREATE_WORKFLOW,
        spec_type=SpecificationType.WORKFLOW,
        example_requests=(
            "Build a pipeline that ingests, transforms, and loads data",
            "Create an ETL workflow from source to destination",
            "Make a data processing pipeline with validation",
        ),
        default_parameters={
            "trigger_type": "manual",
            "steps": ["extract", "transform", "validate", "load"],
            "on_error": "stop",
        },
        tags=("etl", "data", "pipeline"),
    ),

    # --- Agent Templates ---
    IntentTemplate(
        name="monitor_agent",
        description="Agent that monitors systems and alerts on issues",
        intent_type=IntentType.CREATE_AGENT,
        spec_type=SpecificationType.AGENT,
        example_requests=(
            "Create an agent that monitors server health",
            "Build an agent to watch for security alerts",
            "Make an agent that tracks performance metrics",
        ),
        default_parameters={
            "personality_traits": ["vigilant", "analytical"],
            "allowed_actions": ["monitor", "alert", "report"],
        },
        tags=("monitoring", "alerts", "operations"),
    ),
    IntentTemplate(
        name="assistant_agent",
        description="Agent that assists users with specific tasks",
        intent_type=IntentType.CREATE_AGENT,
        spec_type=SpecificationType.AGENT,
        example_requests=(
            "Create a customer support agent",
            "Build a coding assistant agent",
            "Make a research assistant agent",
        ),
        default_parameters={
            "personality_traits": ["helpful", "professional"],
            "memory_enabled": True,
        },
        tags=("assistant", "interactive", "support"),
    ),

    # --- API Templates ---
    IntentTemplate(
        name="crud_api",
        description="REST API with CRUD operations",
        intent_type=IntentType.CREATE_API,
        spec_type=SpecificationType.API,
        example_requests=(
            "Create a REST API for managing users",
            "Build a CRUD API for products",
            "Make an API for blog posts",
        ),
        default_parameters={
            "endpoints": ["list", "create", "read", "update", "delete"],
            "auth_type": "bearer",
        },
        tags=("rest", "crud", "api"),
    ),

    # --- Integration Templates ---
    IntentTemplate(
        name="sync_integration",
        description="Integration that syncs data between systems",
        intent_type=IntentType.CREATE_INTEGRATION,
        spec_type=SpecificationType.INTEGRATION,
        example_requests=(
            "Sync data from Salesforce to our database",
            "Connect Slack with Jira for ticket updates",
            "Integrate GitHub issues with Trello boards",
        ),
        default_parameters={
            "sync_mode": "incremental",
            "on_conflict": "update",
        },
        tags=("sync", "integration", "data"),
    ),
]


class IntentTemplateLibrary:
    """Library of intent templates for matching and suggestion."""

    def __init__(self) -> None:
        self._templates = list(_TEMPLATES)

    def find_matching(
        self,
        intent_type: Optional[IntentType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[IntentTemplate]:
        """Find templates matching criteria."""
        results = self._templates

        if intent_type:
            results = [t for t in results if t.intent_type == intent_type]

        if tags:
            tag_set = set(tags)
            results = [t for t in results if tag_set & set(t.tags)]

        return results[:limit]

    def find_similar(self, text: str, limit: int = 3) -> List[IntentTemplate]:
        """Find templates with similar example requests."""
        text_lower = text.lower()
        scored = []

        for template in self._templates:
            score = 0.0
            for example in template.example_requests:
                # Simple word overlap scoring
                text_words = set(text_lower.split())
                example_words = set(example.lower().split())
                overlap = len(text_words & example_words)
                total = len(text_words | example_words)
                if total > 0:
                    score = max(score, overlap / total)

            if score > 0.1:
                scored.append((score, template))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:limit]]

    def get_examples_for_type(self, intent_type: IntentType) -> List[str]:
        """Get example requests for an intent type."""
        examples: List[str] = []
        for template in self._templates:
            if template.intent_type == intent_type:
                examples.extend(template.example_requests)
        return examples

    def add_template(self, template: IntentTemplate) -> None:
        """Add a custom template."""
        self._templates.append(template)

    @property
    def all_templates(self) -> List[IntentTemplate]:
        return list(self._templates)
