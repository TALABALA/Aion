"""
AION Intent Classification

Classifies user intents for routing and handling decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import re

import structlog

logger = structlog.get_logger(__name__)


class IntentType(str, Enum):
    """Types of user intents."""
    QUESTION = "question"
    COMMAND = "command"
    TASK = "task"
    CONVERSATION = "conversation"
    CLARIFICATION = "clarification"
    FOLLOWUP = "followup"
    FEEDBACK = "feedback"
    GREETING = "greeting"
    FAREWELL = "farewell"
    HELP = "help"
    CANCEL = "cancel"
    CONFIRM = "confirm"
    UNKNOWN = "unknown"


class TaskComplexity(str, Enum):
    """Complexity levels for tasks."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class Intent:
    """Classified intent of a user message."""
    type: IntentType
    confidence: float
    complexity: TaskComplexity = TaskComplexity.SIMPLE

    requires_tools: bool = False
    requires_memory: bool = False
    requires_planning: bool = False

    suggested_tools: list[str] = field(default_factory=list)

    entities: dict[str, Any] = field(default_factory=dict)

    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "confidence": self.confidence,
            "complexity": self.complexity.value,
            "requires_tools": self.requires_tools,
            "requires_memory": self.requires_memory,
            "requires_planning": self.requires_planning,
            "suggested_tools": self.suggested_tools,
            "entities": self.entities,
        }


class IntentClassifier:
    """
    Classifies user intents from messages.

    Uses pattern matching and heuristics for fast classification.
    Can be enhanced with ML-based classification.
    """

    def __init__(self):
        self._patterns = self._build_patterns()
        self._tool_keywords = self._build_tool_keywords()

    def classify(
        self,
        message: str,
        conversation_context: Optional[list[dict]] = None,
    ) -> Intent:
        """
        Classify the intent of a user message.

        Args:
            message: The user's message
            conversation_context: Previous messages for context

        Returns:
            Classified Intent
        """
        message_lower = message.lower().strip()

        for intent_type, patterns in self._patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    intent = self._create_intent(intent_type, message, 0.8)
                    self._enrich_intent(intent, message, conversation_context)
                    return intent

        intent = self._classify_by_structure(message)
        self._enrich_intent(intent, message, conversation_context)

        return intent

    def _create_intent(
        self,
        intent_type: IntentType,
        message: str,
        confidence: float,
    ) -> Intent:
        """Create an Intent object."""
        return Intent(
            type=intent_type,
            confidence=confidence,
            raw_text=message,
        )

    def _classify_by_structure(self, message: str) -> Intent:
        """Classify by message structure when no pattern matches."""
        message_lower = message.lower().strip()

        if message.endswith("?"):
            return self._create_intent(IntentType.QUESTION, message, 0.7)

        imperative_starters = [
            "create", "make", "build", "write", "generate", "find", "search",
            "show", "list", "get", "fetch", "run", "execute", "analyze",
            "calculate", "convert", "compare", "explain", "describe",
            "summarize", "translate", "help", "tell", "give",
        ]

        first_word = message_lower.split()[0] if message_lower else ""
        if first_word in imperative_starters:
            return self._create_intent(IntentType.COMMAND, message, 0.7)

        if len(message.split()) > 20:
            return self._create_intent(IntentType.TASK, message, 0.6)

        return self._create_intent(IntentType.CONVERSATION, message, 0.5)

    def _enrich_intent(
        self,
        intent: Intent,
        message: str,
        context: Optional[list[dict]],
    ) -> None:
        """Enrich intent with additional analysis."""
        message_lower = message.lower()

        for tool, keywords in self._tool_keywords.items():
            if any(kw in message_lower for kw in keywords):
                intent.requires_tools = True
                intent.suggested_tools.append(tool)

        memory_keywords = [
            "remember", "recall", "previous", "earlier", "before",
            "last time", "you said", "we discussed", "history",
        ]
        if any(kw in message_lower for kw in memory_keywords):
            intent.requires_memory = True

        planning_indicators = [
            "step by step", "plan", "how to", "process", "workflow",
            "multiple steps", "first", "then", "after that", "finally",
        ]
        if any(ind in message_lower for ind in planning_indicators):
            intent.requires_planning = True

        intent.complexity = self._assess_complexity(message, intent)

        intent.entities = self._extract_entities(message)

    def _assess_complexity(self, message: str, intent: Intent) -> TaskComplexity:
        """Assess task complexity."""
        complexity_score = 0

        word_count = len(message.split())
        if word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1

        if intent.requires_tools:
            complexity_score += 1
        if intent.requires_planning:
            complexity_score += 2
        if len(intent.suggested_tools) > 2:
            complexity_score += 1

        if complexity_score >= 4:
            return TaskComplexity.COMPLEX
        elif complexity_score >= 2:
            return TaskComplexity.MODERATE
        return TaskComplexity.SIMPLE

    def _extract_entities(self, message: str) -> dict[str, Any]:
        """Extract entities from message."""
        entities: dict[str, Any] = {}

        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, message)
        if urls:
            entities["urls"] = urls

        file_pattern = r'\b[\w\-]+\.[a-zA-Z]{2,4}\b'
        files = re.findall(file_pattern, message)
        if files:
            entities["files"] = files

        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, message)
        if numbers:
            entities["numbers"] = [float(n) if "." in n else int(n) for n in numbers]

        return entities

    def _build_patterns(self) -> dict[IntentType, list[str]]:
        """Build regex patterns for intent classification."""
        return {
            IntentType.GREETING: [
                r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
                r'^(howdy|sup|yo)\b',
            ],
            IntentType.FAREWELL: [
                r'^(bye|goodbye|see you|farewell|take care)\b',
                r'^(good night|later|ciao)\b',
            ],
            IntentType.HELP: [
                r'^(help|how do i|what can you|show me how)\b',
                r'\bhelp me\b',
                r'\bwhat can you do\b',
            ],
            IntentType.CANCEL: [
                r'^(cancel|stop|abort|nevermind|never mind)\b',
                r'\bcancel that\b',
            ],
            IntentType.CONFIRM: [
                r'^(yes|yeah|yep|sure|okay|ok|confirm|proceed)\b',
                r'^(go ahead|do it|sounds good)\b',
            ],
            IntentType.CLARIFICATION: [
                r'^(what do you mean|can you clarify|i don\'t understand)\b',
                r'\bwhat\?\b',
                r'\bhuh\?\b',
            ],
            IntentType.FOLLOWUP: [
                r'^(and |also |additionally |furthermore )',
                r'^(what about|how about|can you also)\b',
            ],
            IntentType.FEEDBACK: [
                r'^(thanks|thank you|great|awesome|perfect)\b',
                r'\bthat\'s (wrong|incorrect|not right)\b',
                r'\bthat\'s (right|correct|perfect)\b',
            ],
        }

    def _build_tool_keywords(self) -> dict[str, list[str]]:
        """Build keyword mappings for tool suggestions."""
        return {
            "web_search": [
                "search", "find online", "look up", "google", "web",
                "internet", "latest", "current", "news",
            ],
            "file_operations": [
                "file", "read", "write", "save", "load", "open",
                "directory", "folder", "path",
            ],
            "code_execution": [
                "run", "execute", "code", "script", "program",
                "python", "javascript", "compile",
            ],
            "image_analysis": [
                "image", "picture", "photo", "screenshot", "visual",
                "look at", "analyze image", "describe image",
            ],
            "calculation": [
                "calculate", "compute", "math", "sum", "average",
                "percentage", "convert", "formula",
            ],
            "data_analysis": [
                "analyze", "data", "statistics", "chart", "graph",
                "visualization", "report", "metrics",
            ],
        }


class IntentRouter:
    """
    Routes requests based on classified intent.
    """

    def __init__(self, classifier: Optional[IntentClassifier] = None):
        self.classifier = classifier or IntentClassifier()

    def get_routing_decision(
        self,
        intent: Intent,
    ) -> dict[str, Any]:
        """
        Get routing decision based on intent.

        Returns:
            Dict with routing information
        """
        decision = {
            "use_tools": intent.requires_tools,
            "use_memory": intent.requires_memory,
            "use_planning": intent.requires_planning,
            "suggested_tools": intent.suggested_tools,
            "priority": self._get_priority(intent),
            "should_stream": True,
        }

        if intent.type in (IntentType.GREETING, IntentType.FAREWELL):
            decision["use_tools"] = False
            decision["use_planning"] = False
            decision["should_stream"] = False

        if intent.type == IntentType.CANCEL:
            decision["action"] = "cancel_current"

        if intent.complexity == TaskComplexity.COMPLEX:
            decision["use_planning"] = True

        return decision

    def _get_priority(self, intent: Intent) -> str:
        """Determine priority based on intent."""
        if intent.type in (IntentType.CANCEL, IntentType.HELP):
            return "high"

        if intent.complexity == TaskComplexity.COMPLEX:
            return "normal"

        if intent.type in (IntentType.GREETING, IntentType.FAREWELL, IntentType.FEEDBACK):
            return "low"

        return "normal"
