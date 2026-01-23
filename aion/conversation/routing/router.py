"""
AION Request Router

Routes conversation requests to appropriate handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import structlog

from aion.conversation.types import (
    Message,
    Conversation,
    ConversationConfig,
)
from aion.conversation.routing.intent import Intent, IntentClassifier, IntentType, IntentRouter

logger = structlog.get_logger(__name__)


@dataclass
class RoutingDecision:
    """Decision about how to handle a request."""
    use_llm: bool = True
    use_tools: bool = False
    use_memory: bool = False
    use_planning: bool = False

    streaming: bool = True

    suggested_tools: list[str] = field(default_factory=list)

    preprocessing: list[str] = field(default_factory=list)
    postprocessing: list[str] = field(default_factory=list)

    special_handler: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_llm": self.use_llm,
            "use_tools": self.use_tools,
            "use_memory": self.use_memory,
            "use_planning": self.use_planning,
            "streaming": self.streaming,
            "suggested_tools": self.suggested_tools,
            "preprocessing": self.preprocessing,
            "postprocessing": self.postprocessing,
            "special_handler": self.special_handler,
            "metadata": self.metadata,
        }


class RequestRouter:
    """
    Routes requests to appropriate processing paths.

    Features:
    - Intent-based routing
    - Configuration-aware decisions
    - Special handler support
    - Preprocessing/postprocessing chains
    """

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
    ):
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.intent_router = IntentRouter(self.intent_classifier)

        self._special_handlers: dict[str, Callable] = {}
        self._preprocessors: dict[str, Callable] = {}
        self._postprocessors: dict[str, Callable] = {}

    def route(
        self,
        message: str,
        conversation: Conversation,
        config: Optional[ConversationConfig] = None,
    ) -> RoutingDecision:
        """
        Determine how to route a request.

        Args:
            message: The user's message
            conversation: The conversation context
            config: Configuration overrides

        Returns:
            RoutingDecision with handling instructions
        """
        config = config or conversation.config

        context = [
            {"role": m.role.value, "content": m.get_text()}
            for m in conversation.messages[-5:]
        ]

        intent = self.intent_classifier.classify(message, context)

        intent_decision = self.intent_router.get_routing_decision(intent)

        decision = RoutingDecision(
            use_llm=True,
            use_tools=intent_decision.get("use_tools", False) and config.tools_enabled,
            use_memory=intent_decision.get("use_memory", False) and config.memory_enabled,
            use_planning=intent_decision.get("use_planning", False) and config.planning_enabled,
            streaming=intent_decision.get("should_stream", True) and config.streaming_enabled,
            suggested_tools=intent_decision.get("suggested_tools", []),
            metadata={
                "intent": intent.to_dict(),
                "intent_decision": intent_decision,
            },
        )

        self._apply_special_handlers(decision, intent, message)

        self._apply_preprocessing(decision, intent, message)

        self._apply_postprocessing(decision, intent)

        logger.debug(
            "Routed request",
            intent=intent.type.value,
            complexity=intent.complexity.value,
            use_tools=decision.use_tools,
            use_memory=decision.use_memory,
            use_planning=decision.use_planning,
        )

        return decision

    def register_special_handler(
        self,
        name: str,
        handler: Callable,
        trigger_intents: Optional[list[IntentType]] = None,
    ) -> None:
        """
        Register a special handler.

        Special handlers bypass normal LLM processing.
        """
        self._special_handlers[name] = {
            "handler": handler,
            "trigger_intents": trigger_intents or [],
        }

    def register_preprocessor(
        self,
        name: str,
        preprocessor: Callable,
    ) -> None:
        """Register a preprocessing function."""
        self._preprocessors[name] = preprocessor

    def register_postprocessor(
        self,
        name: str,
        postprocessor: Callable,
    ) -> None:
        """Register a postprocessing function."""
        self._postprocessors[name] = postprocessor

    def _apply_special_handlers(
        self,
        decision: RoutingDecision,
        intent: Intent,
        message: str,
    ) -> None:
        """Check and apply special handlers."""
        for name, config in self._special_handlers.items():
            trigger_intents = config.get("trigger_intents", [])
            if not trigger_intents or intent.type in trigger_intents:
                decision.special_handler = name
                decision.use_llm = False
                return

        if intent.type == IntentType.GREETING:
            decision.metadata["quick_response"] = True

        if intent.type == IntentType.FAREWELL:
            decision.metadata["quick_response"] = True

    def _apply_preprocessing(
        self,
        decision: RoutingDecision,
        intent: Intent,
        message: str,
    ) -> None:
        """Determine preprocessing steps."""
        if intent.requires_memory:
            decision.preprocessing.append("memory_retrieval")

        if any(url for url in intent.entities.get("urls", [])):
            decision.preprocessing.append("url_expansion")

        if intent.entities.get("files"):
            decision.preprocessing.append("file_analysis")

    def _apply_postprocessing(
        self,
        decision: RoutingDecision,
        intent: Intent,
    ) -> None:
        """Determine postprocessing steps."""
        if decision.use_tools:
            decision.postprocessing.append("tool_result_formatting")

        if intent.type == IntentType.QUESTION:
            decision.postprocessing.append("answer_verification")


class RoutingPipeline:
    """
    Pipeline for request routing with multiple stages.
    """

    def __init__(self, router: RequestRouter):
        self.router = router
        self._stages: list[Callable] = []

    def add_stage(self, stage: Callable) -> "RoutingPipeline":
        """Add a routing stage."""
        self._stages.append(stage)
        return self

    async def process(
        self,
        message: str,
        conversation: Conversation,
    ) -> RoutingDecision:
        """Process through the routing pipeline."""
        decision = self.router.route(message, conversation)

        for stage in self._stages:
            decision = await stage(decision, message, conversation)

        return decision


def create_default_router() -> RequestRouter:
    """Create a router with default configuration."""
    router = RequestRouter()

    return router
