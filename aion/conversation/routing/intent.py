"""
AION SOTA Intent Classification

State-of-the-art intent classification using:
- LLM-based classification with few-shot prompting
- Semantic vector matching with embedding similarity
- Hybrid ensemble approach for maximum accuracy
- Confidence calibration and uncertainty quantification
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

import numpy as np
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
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    EXPLANATION = "explanation"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    PLANNING = "planning"
    UNKNOWN = "unknown"


class TaskComplexity(str, Enum):
    """Complexity levels for tasks."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class UncertaintyLevel(str, Enum):
    """Uncertainty levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class IntentPrediction:
    """A single intent prediction with confidence."""
    intent_type: IntentType
    confidence: float
    reasoning: str = ""


@dataclass
class Intent:
    """Classified intent of a user message with full analysis."""
    type: IntentType
    confidence: float
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    uncertainty: UncertaintyLevel = UncertaintyLevel.MEDIUM

    # Capability requirements
    requires_tools: bool = False
    requires_memory: bool = False
    requires_planning: bool = False
    requires_code_execution: bool = False
    requires_web_access: bool = False
    requires_file_access: bool = False
    requires_vision: bool = False

    # Multi-label predictions (top-k intents)
    all_predictions: list[IntentPrediction] = field(default_factory=list)

    # Tool suggestions with confidence
    suggested_tools: list[tuple[str, float]] = field(default_factory=list)

    # Extracted entities with types
    entities: dict[str, list[Any]] = field(default_factory=dict)

    # Semantic features
    embedding: Optional[list[float]] = None
    semantic_similarity_scores: dict[str, float] = field(default_factory=dict)

    # Analysis metadata
    raw_text: str = ""
    classification_method: str = "ensemble"
    classification_latency_ms: float = 0.0
    llm_reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "confidence": self.confidence,
            "complexity": self.complexity.value,
            "uncertainty": self.uncertainty.value,
            "requires_tools": self.requires_tools,
            "requires_memory": self.requires_memory,
            "requires_planning": self.requires_planning,
            "requires_code_execution": self.requires_code_execution,
            "requires_web_access": self.requires_web_access,
            "requires_file_access": self.requires_file_access,
            "requires_vision": self.requires_vision,
            "all_predictions": [
                {"type": p.intent_type.value, "confidence": p.confidence, "reasoning": p.reasoning}
                for p in self.all_predictions
            ],
            "suggested_tools": self.suggested_tools,
            "entities": self.entities,
            "semantic_similarity_scores": self.semantic_similarity_scores,
            "classification_method": self.classification_method,
            "classification_latency_ms": self.classification_latency_ms,
            "llm_reasoning": self.llm_reasoning,
        }

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.85 and self.uncertainty == UncertaintyLevel.LOW

    @property
    def needs_clarification(self) -> bool:
        return self.uncertainty in (UncertaintyLevel.HIGH, UncertaintyLevel.VERY_HIGH)


# =============================================================================
# Embedding Provider Protocol
# =============================================================================

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class DefaultEmbeddingProvider:
    """
    Default embedding provider using sentence transformers or API.
    Falls back to TF-IDF if no model available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._tfidf_vectorizer = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._initialized:
            return

        try:
            # Try to load sentence-transformers
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using TF-IDF fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(max_features=384)
            # Fit on some default texts
            default_texts = [
                "hello how are you", "create a function", "search the web",
                "analyze this data", "explain this code", "fix this bug",
                "write a story", "help me plan", "remember what I said",
            ]
            self._tfidf_vectorizer.fit(default_texts)

        self._initialized = True

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if not self._initialized:
            await self.initialize()

        if self._model is not None:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        elif self._tfidf_vectorizer is not None:
            vector = self._tfidf_vectorizer.transform([text]).toarray()[0]
            return vector.tolist()
        else:
            # Ultimate fallback: hash-based pseudo-embedding
            return self._hash_embedding(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            await self.initialize()

        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return [await self.embed(text) for text in texts]

    def _hash_embedding(self, text: str, dim: int = 384) -> list[float]:
        """Generate pseudo-embedding using locality-sensitive hashing."""
        embedding = []
        for i in range(dim):
            h = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
            value = (int(h[:8], 16) / (2**32)) * 2 - 1  # Normalize to [-1, 1]
            embedding.append(value)
        return embedding


# =============================================================================
# LLM-Based Intent Classifier
# =============================================================================

class LLMIntentClassifier:
    """
    LLM-based intent classification using few-shot prompting.
    Provides detailed reasoning and multi-label predictions.
    """

    CLASSIFICATION_PROMPT = """You are an expert intent classifier for an AI assistant. Analyze the user's message and classify their intent.

## Intent Types:
- QUESTION: Asking for information or explanation
- COMMAND: Direct instruction to perform an action
- TASK: Complex multi-step request
- CODE_GENERATION: Request to write code
- CODE_REVIEW: Request to review/analyze code
- DEBUGGING: Request to fix bugs or errors
- EXPLANATION: Request for detailed explanation
- CREATIVE_WRITING: Request for creative content
- DATA_ANALYSIS: Request to analyze data
- RESEARCH: Request for in-depth research
- PLANNING: Request to create plans or strategies
- GREETING: Social greeting
- FAREWELL: Ending conversation
- HELP: Asking for assistance with the system
- CANCEL: Canceling a request
- CONFIRM: Confirming an action
- CLARIFICATION: Asking for clarification
- FOLLOWUP: Following up on previous topic
- FEEDBACK: Providing feedback
- CONVERSATION: General conversation
- UNKNOWN: Cannot determine intent

## Examples:

Message: "Write a Python function to sort a list using quicksort"
Analysis: {
    "primary_intent": "CODE_GENERATION",
    "confidence": 0.95,
    "secondary_intents": [{"type": "EXPLANATION", "confidence": 0.3}],
    "complexity": "MODERATE",
    "requires_code_execution": true,
    "reasoning": "User explicitly asks to write code (Python function) for a specific algorithm."
}

Message: "What's the time complexity of binary search and why?"
Analysis: {
    "primary_intent": "QUESTION",
    "confidence": 0.92,
    "secondary_intents": [{"type": "EXPLANATION", "confidence": 0.85}],
    "complexity": "SIMPLE",
    "requires_code_execution": false,
    "reasoning": "User asks a conceptual question about algorithm complexity with request for explanation."
}

Message: "Hey there!"
Analysis: {
    "primary_intent": "GREETING",
    "confidence": 0.98,
    "secondary_intents": [],
    "complexity": "TRIVIAL",
    "requires_code_execution": false,
    "reasoning": "Simple social greeting with no actionable request."
}

Message: "Can you search for the latest research on transformer architectures and summarize the key findings?"
Analysis: {
    "primary_intent": "RESEARCH",
    "confidence": 0.88,
    "secondary_intents": [{"type": "TASK", "confidence": 0.7}, {"type": "DATA_ANALYSIS", "confidence": 0.4}],
    "complexity": "COMPLEX",
    "requires_web_access": true,
    "reasoning": "Multi-step request requiring web search, reading multiple sources, and synthesis."
}

Now analyze this message:

Message: "{message}"

{context_section}

Respond with ONLY a valid JSON object in the exact format shown above."""

    def __init__(self, llm_provider: Any = None):
        self.llm_provider = llm_provider
        self._cache: dict[str, dict] = {}
        self._cache_ttl = 3600  # 1 hour

    async def classify(
        self,
        message: str,
        conversation_context: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """
        Classify intent using LLM with few-shot prompting.

        Returns parsed classification result.
        """
        # Check cache
        cache_key = hashlib.md5(message.encode()).hexdigest()
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached["timestamp"] < self._cache_ttl:
                return cached["result"]

        # Build context section
        context_section = ""
        if conversation_context:
            recent = conversation_context[-3:]  # Last 3 messages
            context_section = "Recent conversation context:\n"
            for msg in recent:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                context_section += f"- {role}: {content}\n"

        prompt = self.CLASSIFICATION_PROMPT.format(
            message=message,
            context_section=context_section,
        )

        try:
            if self.llm_provider:
                response = await self.llm_provider.complete(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1,  # Low temperature for consistency
                )
                result_text = response.content
            else:
                # Fallback to rule-based if no LLM
                return self._rule_based_fallback(message)

            # Parse JSON response
            result = self._parse_llm_response(result_text)

            # Cache result
            self._cache[cache_key] = {
                "result": result,
                "timestamp": time.time(),
            }

            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._rule_based_fallback(message)

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Return default if parsing fails
        return {
            "primary_intent": "UNKNOWN",
            "confidence": 0.5,
            "secondary_intents": [],
            "complexity": "SIMPLE",
            "reasoning": "Failed to parse LLM response",
        }

    def _rule_based_fallback(self, message: str) -> dict[str, Any]:
        """Fallback to rule-based classification."""
        message_lower = message.lower().strip()

        # Simple pattern matching as fallback
        patterns = {
            "CODE_GENERATION": [r'\b(write|create|implement|code|function|class)\b.*\b(python|javascript|code)\b'],
            "DEBUGGING": [r'\b(fix|debug|error|bug|issue|problem)\b'],
            "QUESTION": [r'\?$', r'^(what|why|how|when|where|who|which)\b'],
            "GREETING": [r'^(hi|hello|hey|greetings)\b'],
            "FAREWELL": [r'^(bye|goodbye|see you)\b'],
        }

        for intent, pats in patterns.items():
            for pat in pats:
                if re.search(pat, message_lower):
                    return {
                        "primary_intent": intent,
                        "confidence": 0.7,
                        "secondary_intents": [],
                        "complexity": "SIMPLE",
                        "reasoning": f"Matched pattern: {pat}",
                    }

        return {
            "primary_intent": "CONVERSATION",
            "confidence": 0.5,
            "secondary_intents": [],
            "complexity": "SIMPLE",
            "reasoning": "No pattern matched, defaulting to conversation",
        }


# =============================================================================
# Semantic Vector Classifier
# =============================================================================

class SemanticIntentClassifier:
    """
    Intent classification using semantic vector similarity.
    Compares message embedding against prototype embeddings for each intent.
    """

    # Prototype examples for each intent type
    INTENT_PROTOTYPES = {
        IntentType.QUESTION: [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Why is the sky blue?",
            "Can you explain quantum computing?",
            "What are the benefits of exercise?",
        ],
        IntentType.COMMAND: [
            "Turn off the lights",
            "Set a reminder for tomorrow",
            "Open the file manager",
            "Delete this message",
            "Save the document",
        ],
        IntentType.CODE_GENERATION: [
            "Write a Python function to calculate fibonacci",
            "Create a React component for a login form",
            "Implement a binary search tree in Java",
            "Write SQL query to join two tables",
            "Create a REST API endpoint in Node.js",
        ],
        IntentType.CODE_REVIEW: [
            "Review this code for best practices",
            "Check this function for bugs",
            "Analyze the performance of this algorithm",
            "Is this code following SOLID principles?",
            "What improvements can be made to this class?",
        ],
        IntentType.DEBUGGING: [
            "Fix this error in my code",
            "Why is this function returning null?",
            "Debug this segmentation fault",
            "Help me solve this exception",
            "My code is crashing, can you help?",
        ],
        IntentType.EXPLANATION: [
            "Explain how neural networks work",
            "Walk me through this algorithm step by step",
            "Break down this complex concept",
            "Help me understand recursion",
            "Explain the difference between REST and GraphQL",
        ],
        IntentType.CREATIVE_WRITING: [
            "Write a short story about space exploration",
            "Create a poem about nature",
            "Help me write a blog post",
            "Draft an email to my team",
            "Write a product description",
        ],
        IntentType.DATA_ANALYSIS: [
            "Analyze this dataset for trends",
            "Create a visualization of sales data",
            "Calculate statistics for these numbers",
            "Find correlations in this data",
            "Generate a report from this CSV",
        ],
        IntentType.RESEARCH: [
            "Research the latest AI developments",
            "Find information about climate change",
            "Look up best practices for microservices",
            "Investigate competitor pricing strategies",
            "Gather data on market trends",
        ],
        IntentType.PLANNING: [
            "Create a project plan for the app launch",
            "Help me plan my week",
            "Design a roadmap for the product",
            "Outline a strategy for growth",
            "Plan the implementation steps",
        ],
        IntentType.GREETING: [
            "Hello!",
            "Hi there",
            "Good morning",
            "Hey, how are you?",
            "Greetings",
        ],
        IntentType.FAREWELL: [
            "Goodbye",
            "See you later",
            "Bye for now",
            "Take care",
            "Until next time",
        ],
        IntentType.HELP: [
            "How do I use this?",
            "What can you do?",
            "I need help",
            "Show me how to get started",
            "What features do you have?",
        ],
        IntentType.CANCEL: [
            "Cancel that",
            "Never mind",
            "Stop",
            "Abort the operation",
            "Don't do that",
        ],
        IntentType.CONFIRM: [
            "Yes, proceed",
            "Go ahead",
            "Confirmed",
            "That's correct",
            "Do it",
        ],
    }

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        self.embedding_provider = embedding_provider or DefaultEmbeddingProvider()
        self._prototype_embeddings: dict[IntentType, list[list[float]]] = {}
        self._prototype_centroids: dict[IntentType, list[float]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize prototype embeddings."""
        if self._initialized:
            return

        if hasattr(self.embedding_provider, 'initialize'):
            await self.embedding_provider.initialize()

        logger.info("Computing prototype embeddings for intent classification...")

        for intent_type, examples in self.INTENT_PROTOTYPES.items():
            embeddings = await self.embedding_provider.embed_batch(examples)
            self._prototype_embeddings[intent_type] = embeddings

            # Compute centroid
            centroid = np.mean(embeddings, axis=0).tolist()
            self._prototype_centroids[intent_type] = centroid

        self._initialized = True
        logger.info(f"Initialized semantic classifier with {len(self._prototype_centroids)} intent types")

    async def classify(
        self,
        message: str,
        top_k: int = 3,
    ) -> list[tuple[IntentType, float]]:
        """
        Classify intent using semantic similarity.

        Returns top-k intents with similarity scores.
        """
        if not self._initialized:
            await self.initialize()

        # Get message embedding
        message_embedding = await self.embedding_provider.embed(message)
        message_np = np.array(message_embedding)

        # Compute similarity to each intent centroid
        similarities = {}
        for intent_type, centroid in self._prototype_centroids.items():
            centroid_np = np.array(centroid)
            similarity = self._cosine_similarity(message_np, centroid_np)
            similarities[intent_type] = similarity

        # Sort by similarity
        sorted_intents = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_intents[:top_k]

    async def get_embedding(self, message: str) -> list[float]:
        """Get embedding for a message."""
        if not self._initialized:
            await self.initialize()
        return await self.embedding_provider.embed(message)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# Ensemble Intent Classifier (SOTA)
# =============================================================================

class EnsembleIntentClassifier:
    """
    State-of-the-art ensemble intent classifier combining:
    - LLM-based classification (understanding + reasoning)
    - Semantic vector similarity (fast + robust)
    - Rule-based patterns (deterministic fallback)

    Uses confidence calibration and uncertainty quantification.
    """

    def __init__(
        self,
        llm_provider: Any = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_weight: float = 0.5,
        semantic_weight: float = 0.35,
        rule_weight: float = 0.15,
    ):
        self.llm_classifier = LLMIntentClassifier(llm_provider)
        self.semantic_classifier = SemanticIntentClassifier(embedding_provider)
        self.rule_classifier = RuleBasedClassifier()

        # Ensemble weights
        self.llm_weight = llm_weight
        self.semantic_weight = semantic_weight
        self.rule_weight = rule_weight

        # Entity extractor
        self.entity_extractor = EntityExtractor()

        # Tool suggester
        self.tool_suggester = ToolSuggester()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all classifiers."""
        if self._initialized:
            return

        await self.semantic_classifier.initialize()
        self._initialized = True
        logger.info("Ensemble intent classifier initialized")

    async def classify(
        self,
        message: str,
        conversation_context: Optional[list[dict]] = None,
    ) -> Intent:
        """
        Classify intent using ensemble of methods.

        Args:
            message: User message to classify
            conversation_context: Previous messages for context

        Returns:
            Comprehensive Intent object
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Run classifiers in parallel
        llm_task = asyncio.create_task(
            self.llm_classifier.classify(message, conversation_context)
        )
        semantic_task = asyncio.create_task(
            self.semantic_classifier.classify(message, top_k=5)
        )

        llm_result = await llm_task
        semantic_results = await semantic_task
        rule_result = self.rule_classifier.classify(message)

        # Get embedding for the message
        embedding = await self.semantic_classifier.get_embedding(message)

        # Combine predictions
        intent_scores: dict[IntentType, float] = {}

        # Add LLM predictions
        primary_intent = IntentType(llm_result.get("primary_intent", "UNKNOWN"))
        llm_confidence = llm_result.get("confidence", 0.5)
        intent_scores[primary_intent] = intent_scores.get(primary_intent, 0) + \
            self.llm_weight * llm_confidence

        for secondary in llm_result.get("secondary_intents", []):
            sec_intent = IntentType(secondary.get("type", "UNKNOWN"))
            sec_conf = secondary.get("confidence", 0.3)
            intent_scores[sec_intent] = intent_scores.get(sec_intent, 0) + \
                self.llm_weight * sec_conf * 0.5  # Discount secondary

        # Add semantic predictions
        for intent_type, similarity in semantic_results:
            # Convert similarity to confidence (0.5-1.0 range is meaningful)
            confidence = max(0, (similarity - 0.3) / 0.7)
            intent_scores[intent_type] = intent_scores.get(intent_type, 0) + \
                self.semantic_weight * confidence

        # Add rule-based predictions
        intent_scores[rule_result["intent"]] = intent_scores.get(rule_result["intent"], 0) + \
            self.rule_weight * rule_result["confidence"]

        # Normalize scores
        total_score = sum(intent_scores.values())
        if total_score > 0:
            for intent in intent_scores:
                intent_scores[intent] /= total_score

        # Get top prediction
        sorted_predictions = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_intent, top_confidence = sorted_predictions[0]

        # Build all predictions list
        all_predictions = [
            IntentPrediction(
                intent_type=intent,
                confidence=conf,
                reasoning="",
            )
            for intent, conf in sorted_predictions[:5]
        ]

        # Determine complexity
        complexity = self._map_complexity(llm_result.get("complexity", "SIMPLE"))

        # Quantify uncertainty
        uncertainty = self._calculate_uncertainty(sorted_predictions)

        # Extract entities
        entities = self.entity_extractor.extract(message)

        # Suggest tools
        suggested_tools = self.tool_suggester.suggest(message, top_intent, entities)

        # Build semantic similarity scores
        semantic_similarity_scores = {
            intent_type.value: score
            for intent_type, score in semantic_results
        }

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Build Intent object
        intent = Intent(
            type=top_intent,
            confidence=top_confidence,
            complexity=complexity,
            uncertainty=uncertainty,
            requires_tools=llm_result.get("requires_tools", False) or bool(suggested_tools),
            requires_memory=llm_result.get("requires_memory", False) or self._needs_memory(message),
            requires_planning=llm_result.get("requires_planning", False) or complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT),
            requires_code_execution=llm_result.get("requires_code_execution", False),
            requires_web_access=llm_result.get("requires_web_access", False) or self._needs_web(message),
            requires_file_access=llm_result.get("requires_file_access", False) or self._needs_files(message),
            requires_vision=llm_result.get("requires_vision", False) or self._needs_vision(message),
            all_predictions=all_predictions,
            suggested_tools=suggested_tools,
            entities=entities,
            embedding=embedding,
            semantic_similarity_scores=semantic_similarity_scores,
            raw_text=message,
            classification_method="ensemble",
            classification_latency_ms=latency_ms,
            llm_reasoning=llm_result.get("reasoning", ""),
        )

        logger.debug(
            "Intent classified",
            intent=intent.type.value,
            confidence=f"{intent.confidence:.2f}",
            complexity=intent.complexity.value,
            latency_ms=f"{latency_ms:.1f}",
        )

        return intent

    def _map_complexity(self, complexity_str: str) -> TaskComplexity:
        """Map string to TaskComplexity enum."""
        mapping = {
            "TRIVIAL": TaskComplexity.TRIVIAL,
            "SIMPLE": TaskComplexity.SIMPLE,
            "MODERATE": TaskComplexity.MODERATE,
            "COMPLEX": TaskComplexity.COMPLEX,
            "EXPERT": TaskComplexity.EXPERT,
        }
        return mapping.get(complexity_str.upper(), TaskComplexity.SIMPLE)

    def _calculate_uncertainty(
        self,
        predictions: list[tuple[IntentType, float]],
    ) -> UncertaintyLevel:
        """Calculate uncertainty based on prediction distribution."""
        if len(predictions) < 2:
            return UncertaintyLevel.HIGH

        top_conf = predictions[0][1]
        second_conf = predictions[1][1]

        # Confidence gap between top two predictions
        gap = top_conf - second_conf

        # Entropy-based uncertainty
        confs = [p[1] for p in predictions if p[1] > 0.01]
        entropy = -sum(c * np.log(c + 1e-10) for c in confs)
        max_entropy = np.log(len(confs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        if top_conf >= 0.85 and gap >= 0.3:
            return UncertaintyLevel.LOW
        elif top_conf >= 0.6 and gap >= 0.15:
            return UncertaintyLevel.MEDIUM
        elif top_conf >= 0.4 or normalized_entropy < 0.7:
            return UncertaintyLevel.HIGH
        else:
            return UncertaintyLevel.VERY_HIGH

    def _needs_memory(self, message: str) -> bool:
        """Check if message likely needs memory access."""
        memory_indicators = [
            r'\b(remember|recall|previous|earlier|before|last time)\b',
            r'\b(you said|we discussed|mentioned|told me)\b',
            r'\b(history|context|conversation)\b',
        ]
        message_lower = message.lower()
        return any(re.search(p, message_lower) for p in memory_indicators)

    def _needs_web(self, message: str) -> bool:
        """Check if message likely needs web access."""
        web_indicators = [
            r'\b(search|google|look up|find online|web|internet)\b',
            r'\b(latest|current|recent|today|news)\b',
            r'\b(website|url|link)\b',
        ]
        message_lower = message.lower()
        return any(re.search(p, message_lower) for p in web_indicators)

    def _needs_files(self, message: str) -> bool:
        """Check if message likely needs file access."""
        file_indicators = [
            r'\b(file|folder|directory|path)\b',
            r'\b(read|write|save|load|open)\b.*\b(file|document)\b',
            r'\.(txt|pdf|csv|json|xml|py|js|ts|md)\b',
        ]
        message_lower = message.lower()
        return any(re.search(p, message_lower) for p in file_indicators)

    def _needs_vision(self, message: str) -> bool:
        """Check if message likely needs vision capability."""
        vision_indicators = [
            r'\b(image|picture|photo|screenshot|diagram)\b',
            r'\b(look at|see|view|show me)\b',
            r'\b(visual|visually|appearance)\b',
        ]
        message_lower = message.lower()
        return any(re.search(p, message_lower) for p in vision_indicators)


# =============================================================================
# Supporting Classes
# =============================================================================

class RuleBasedClassifier:
    """Fast rule-based classifier for common patterns."""

    def __init__(self):
        self._patterns = self._build_patterns()

    def classify(self, message: str) -> dict[str, Any]:
        """Classify using pattern matching."""
        message_lower = message.lower().strip()

        for intent_type, patterns in self._patterns.items():
            for pattern, confidence in patterns:
                if re.search(pattern, message_lower):
                    return {
                        "intent": intent_type,
                        "confidence": confidence,
                        "pattern": pattern,
                    }

        # Default based on structure
        if message.endswith("?"):
            return {"intent": IntentType.QUESTION, "confidence": 0.6, "pattern": "question_mark"}

        return {"intent": IntentType.CONVERSATION, "confidence": 0.4, "pattern": "default"}

    def _build_patterns(self) -> dict[IntentType, list[tuple[str, float]]]:
        """Build pattern rules with confidence scores."""
        return {
            IntentType.GREETING: [
                (r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b', 0.95),
                (r'^(howdy|sup|yo|hiya)\b', 0.9),
            ],
            IntentType.FAREWELL: [
                (r'^(bye|goodbye|see you|farewell|take care)\b', 0.95),
                (r'^(good night|later|ciao|peace)\b', 0.9),
            ],
            IntentType.HELP: [
                (r'^help\b', 0.9),
                (r'\bwhat can you do\b', 0.85),
                (r'\bhow do i use\b', 0.85),
            ],
            IntentType.CANCEL: [
                (r'^(cancel|stop|abort|nevermind|never mind)\b', 0.95),
            ],
            IntentType.CONFIRM: [
                (r'^(yes|yeah|yep|sure|okay|ok|confirm|proceed|go ahead)\b', 0.9),
            ],
            IntentType.CODE_GENERATION: [
                (r'\b(write|create|implement|generate)\b.*\b(code|function|class|script|program)\b', 0.85),
                (r'\b(code|implement)\b.*\b(in python|in javascript|in java|in typescript)\b', 0.85),
            ],
            IntentType.DEBUGGING: [
                (r'\b(fix|debug|error|bug|issue|problem|crash|exception)\b', 0.8),
                (r'\b(not working|doesn\'t work|broken|failing)\b', 0.75),
            ],
        }


class EntityExtractor:
    """Extract entities from messages."""

    def extract(self, message: str) -> dict[str, list[Any]]:
        """Extract various entity types from message."""
        entities: dict[str, list[Any]] = {}

        # URLs
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', message)
        if urls:
            entities["urls"] = urls

        # File paths
        paths = re.findall(r'(?:/[\w.-]+)+|\b[\w.-]+\.[a-zA-Z]{2,4}\b', message)
        if paths:
            entities["file_paths"] = paths

        # Code blocks
        code_blocks = re.findall(r'```[\s\S]*?```|`[^`]+`', message)
        if code_blocks:
            entities["code_blocks"] = code_blocks

        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', message)
        if numbers:
            entities["numbers"] = [float(n) if '.' in n else int(n) for n in numbers]

        # Emails
        emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', message)
        if emails:
            entities["emails"] = emails

        # Programming languages
        languages = re.findall(
            r'\b(python|javascript|typescript|java|c\+\+|rust|go|ruby|php|swift|kotlin)\b',
            message.lower()
        )
        if languages:
            entities["programming_languages"] = list(set(languages))

        # Dates (simple patterns)
        dates = re.findall(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b',
            message.lower()
        )
        if dates:
            entities["dates"] = dates

        return entities


class ToolSuggester:
    """Suggest tools based on intent and entities."""

    TOOL_MAPPINGS = {
        IntentType.CODE_GENERATION: [("code_interpreter", 0.9), ("file_writer", 0.7)],
        IntentType.CODE_REVIEW: [("code_analyzer", 0.9), ("linter", 0.7)],
        IntentType.DEBUGGING: [("debugger", 0.9), ("code_interpreter", 0.8), ("stack_trace_analyzer", 0.7)],
        IntentType.DATA_ANALYSIS: [("data_analyzer", 0.9), ("chart_generator", 0.8), ("statistics", 0.7)],
        IntentType.RESEARCH: [("web_search", 0.95), ("document_reader", 0.8)],
        IntentType.CREATIVE_WRITING: [("text_generator", 0.8), ("grammar_checker", 0.6)],
    }

    ENTITY_TOOL_MAPPINGS = {
        "urls": [("web_fetcher", 0.9), ("link_analyzer", 0.7)],
        "file_paths": [("file_reader", 0.9), ("file_writer", 0.7)],
        "code_blocks": [("code_interpreter", 0.85), ("syntax_highlighter", 0.6)],
        "emails": [("email_sender", 0.7)],
    }

    def suggest(
        self,
        message: str,
        intent: IntentType,
        entities: dict[str, list[Any]],
    ) -> list[tuple[str, float]]:
        """Suggest tools with confidence scores."""
        suggestions: dict[str, float] = {}

        # Add intent-based suggestions
        if intent in self.TOOL_MAPPINGS:
            for tool, confidence in self.TOOL_MAPPINGS[intent]:
                suggestions[tool] = max(suggestions.get(tool, 0), confidence)

        # Add entity-based suggestions
        for entity_type, values in entities.items():
            if entity_type in self.ENTITY_TOOL_MAPPINGS:
                for tool, confidence in self.ENTITY_TOOL_MAPPINGS[entity_type]:
                    suggestions[tool] = max(suggestions.get(tool, 0), confidence)

        # Check for web-related keywords
        if re.search(r'\b(search|google|find online|look up)\b', message.lower()):
            suggestions["web_search"] = max(suggestions.get("web_search", 0), 0.9)

        # Sort by confidence
        sorted_suggestions = sorted(
            suggestions.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_suggestions[:5]


# =============================================================================
# Intent Router (Updated)
# =============================================================================

class IntentRouter:
    """
    Routes requests based on classified intent.
    Uses sophisticated routing logic based on intent analysis.
    """

    def __init__(self, classifier: Optional[EnsembleIntentClassifier] = None):
        self.classifier = classifier or EnsembleIntentClassifier()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the router and classifier."""
        if self._initialized:
            return
        await self.classifier.initialize()
        self._initialized = True

    async def route(
        self,
        message: str,
        conversation_context: Optional[list[dict]] = None,
    ) -> tuple[Intent, dict[str, Any]]:
        """
        Classify intent and determine routing.

        Returns:
            Tuple of (Intent, routing_decision)
        """
        if not self._initialized:
            await self.initialize()

        intent = await self.classifier.classify(message, conversation_context)
        decision = self.get_routing_decision(intent)

        return intent, decision

    def get_routing_decision(self, intent: Intent) -> dict[str, Any]:
        """
        Get routing decision based on intent.

        Returns:
            Dict with routing information
        """
        decision = {
            "use_tools": intent.requires_tools,
            "use_memory": intent.requires_memory,
            "use_planning": intent.requires_planning,
            "use_code_execution": intent.requires_code_execution,
            "use_web_access": intent.requires_web_access,
            "use_file_access": intent.requires_file_access,
            "use_vision": intent.requires_vision,
            "suggested_tools": [t[0] for t in intent.suggested_tools[:3]],
            "priority": self._get_priority(intent),
            "should_stream": self._should_stream(intent),
            "max_tokens": self._get_max_tokens(intent),
            "temperature": self._get_temperature(intent),
            "needs_clarification": intent.needs_clarification,
        }

        # Special handling for certain intents
        if intent.type == IntentType.CANCEL:
            decision["action"] = "cancel_current"
            decision["should_stream"] = False

        if intent.type in (IntentType.GREETING, IntentType.FAREWELL):
            decision["use_tools"] = False
            decision["use_planning"] = False
            decision["should_stream"] = False
            decision["max_tokens"] = 100

        if intent.type == IntentType.CONFIRM:
            decision["action"] = "confirm_pending"

        if intent.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT):
            decision["use_planning"] = True
            decision["max_tokens"] = min(decision["max_tokens"] * 2, 8000)

        return decision

    def _get_priority(self, intent: Intent) -> str:
        """Determine priority based on intent."""
        if intent.type in (IntentType.CANCEL, IntentType.HELP):
            return "high"

        if intent.type == IntentType.DEBUGGING:
            return "high"

        if intent.complexity == TaskComplexity.EXPERT:
            return "high"

        if intent.type in (IntentType.GREETING, IntentType.FAREWELL, IntentType.FEEDBACK):
            return "low"

        return "normal"

    def _should_stream(self, intent: Intent) -> bool:
        """Determine if response should be streamed."""
        # Don't stream short responses
        if intent.type in (IntentType.GREETING, IntentType.FAREWELL, IntentType.CONFIRM, IntentType.CANCEL):
            return False

        # Stream complex tasks
        if intent.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT):
            return True

        # Stream code generation
        if intent.type in (IntentType.CODE_GENERATION, IntentType.EXPLANATION, IntentType.CREATIVE_WRITING):
            return True

        return True  # Default to streaming

    def _get_max_tokens(self, intent: Intent) -> int:
        """Determine max tokens based on intent."""
        token_map = {
            IntentType.GREETING: 100,
            IntentType.FAREWELL: 100,
            IntentType.CONFIRM: 50,
            IntentType.CANCEL: 100,
            IntentType.QUESTION: 1000,
            IntentType.CODE_GENERATION: 4000,
            IntentType.CODE_REVIEW: 2000,
            IntentType.DEBUGGING: 3000,
            IntentType.EXPLANATION: 2000,
            IntentType.CREATIVE_WRITING: 3000,
            IntentType.DATA_ANALYSIS: 2000,
            IntentType.RESEARCH: 4000,
            IntentType.PLANNING: 3000,
            IntentType.TASK: 4000,
        }
        return token_map.get(intent.type, 1500)

    def _get_temperature(self, intent: Intent) -> float:
        """Determine temperature based on intent."""
        temp_map = {
            IntentType.CODE_GENERATION: 0.2,
            IntentType.DEBUGGING: 0.1,
            IntentType.DATA_ANALYSIS: 0.1,
            IntentType.CREATIVE_WRITING: 0.8,
            IntentType.EXPLANATION: 0.4,
            IntentType.QUESTION: 0.3,
            IntentType.RESEARCH: 0.3,
        }
        return temp_map.get(intent.type, 0.5)


# =============================================================================
# Legacy Compatibility
# =============================================================================

# Alias for backwards compatibility
IntentClassifier = EnsembleIntentClassifier
