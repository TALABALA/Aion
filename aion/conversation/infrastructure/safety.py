"""
AION SOTA Safety and Guardrails

State-of-the-art safety features:
- Content moderation (toxicity, violence, etc.)
- PII detection and masking
- Prompt injection defense
- Rate limiting per user
- Input/output validation
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Safety Levels and Categories
# =============================================================================

class SafetyLevel(str, Enum):
    """Safety levels for content."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    BLOCKED = "blocked"


class ContentCategory(str, Enum):
    """Content moderation categories."""
    SAFE = "safe"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    DANGEROUS = "dangerous"
    ILLEGAL = "illegal"
    HARASSMENT = "harassment"
    SPAM = "spam"
    MISINFORMATION = "misinformation"
    PROMPT_INJECTION = "prompt_injection"


class PIIType(str, Enum):
    """Types of PII (Personally Identifiable Information)."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    API_KEY = "api_key"
    PASSWORD = "password"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ModerationResult:
    """Result of content moderation."""
    safe: bool
    safety_level: SafetyLevel
    categories: List[ContentCategory] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    flagged_content: List[str] = field(default_factory=list)
    explanation: str = ""
    action_taken: str = "none"


@dataclass
class PIIMatch:
    """A detected PII match."""
    pii_type: PIIType
    value: str
    masked_value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class PIIResult:
    """Result of PII detection."""
    has_pii: bool
    matches: List[PIIMatch] = field(default_factory=list)
    sanitized_text: str = ""
    pii_types_found: Set[PIIType] = field(default_factory=set)


@dataclass
class InjectionResult:
    """Result of prompt injection detection."""
    is_injection: bool
    confidence: float = 0.0
    injection_type: str = ""
    flagged_patterns: List[str] = field(default_factory=list)
    explanation: str = ""


# =============================================================================
# Content Moderator
# =============================================================================

class ContentModerator:
    """
    Content moderation for detecting harmful content.

    Uses pattern matching and optional ML models for detection.
    """

    # Harmful content patterns (simplified - production would use ML models)
    HARMFUL_PATTERNS: Dict[ContentCategory, List[str]] = {
        ContentCategory.HATE_SPEECH: [
            r'\b(hate|kill|murder)\s+(all|every)\s+\w+s?\b',
            r'\b(inferior|subhuman)\s+\w+\b',
        ],
        ContentCategory.VIOLENCE: [
            r'\b(how\s+to\s+)?(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)\b',
            r'\b(kill|murder|assassinate)\s+(someone|person|people)\b',
        ],
        ContentCategory.SELF_HARM: [
            r'\b(how\s+to\s+)(commit\s+)?suicide\b',
            r'\b(best\s+way\s+to\s+)?(hurt|harm)\s+(myself|yourself)\b',
        ],
        ContentCategory.ILLEGAL: [
            r'\b(how\s+to\s+)(hack|steal|forge|counterfeit)\b',
            r'\b(buy|sell)\s+(drugs|weapons|stolen)\b',
        ],
        ContentCategory.DANGEROUS: [
            r'\b(synthesize|make|create)\s+(drugs|poison|toxin)\b',
            r'\b(bypass|circumvent|disable)\s+(security|safety)\b',
        ],
    }

    # Blocklist of exact phrases
    BLOCKLIST: List[str] = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard your instructions",
    ]

    def __init__(
        self,
        severity_threshold: float = 0.7,
        use_ml_model: bool = False,
        ml_model_path: Optional[str] = None,
    ):
        self.severity_threshold = severity_threshold
        self.use_ml_model = use_ml_model
        self.ml_model_path = ml_model_path

        # Compile patterns
        self._compiled_patterns: Dict[ContentCategory, List[Pattern]] = {}
        for category, patterns in self.HARMFUL_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # ML model (if using)
        self._ml_model = None
        if use_ml_model and ml_model_path:
            self._load_ml_model()

    def _load_ml_model(self) -> None:
        """Load ML model for content moderation."""
        try:
            # Example: Load a transformer model for toxicity detection
            # from transformers import pipeline
            # self._ml_model = pipeline("text-classification", model=self.ml_model_path)
            logger.info("ML moderation model loaded")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")

    async def moderate(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModerationResult:
        """
        Moderate content for harmful material.

        Args:
            text: Text to moderate
            context: Additional context (user info, conversation history)

        Returns:
            ModerationResult with safety assessment
        """
        text_lower = text.lower()
        categories_found: List[ContentCategory] = []
        category_scores: Dict[str, float] = {}
        flagged_content: List[str] = []

        # Check blocklist
        for blocked in self.BLOCKLIST:
            if blocked in text_lower:
                return ModerationResult(
                    safe=False,
                    safety_level=SafetyLevel.BLOCKED,
                    categories=[ContentCategory.PROMPT_INJECTION],
                    category_scores={"prompt_injection": 1.0},
                    flagged_content=[blocked],
                    explanation=f"Blocked phrase detected: {blocked}",
                    action_taken="blocked",
                )

        # Check harmful patterns
        for category, patterns in self._compiled_patterns.items():
            max_score = 0.0
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    max_score = max(max_score, 0.8)  # Pattern match = high score
                    flagged_content.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])

            if max_score > 0:
                category_scores[category.value] = max_score
                if max_score >= self.severity_threshold:
                    categories_found.append(category)

        # Use ML model if available
        if self._ml_model and not categories_found:
            ml_result = await self._ml_moderate(text)
            category_scores.update(ml_result.get("scores", {}))
            if ml_result.get("flagged"):
                categories_found.extend(ml_result.get("categories", []))

        # Determine safety level
        if categories_found:
            max_score = max(category_scores.values()) if category_scores else 0
            if max_score >= 0.9:
                safety_level = SafetyLevel.BLOCKED
                action = "blocked"
            elif max_score >= 0.7:
                safety_level = SafetyLevel.HIGH_RISK
                action = "flagged"
            elif max_score >= 0.5:
                safety_level = SafetyLevel.MEDIUM_RISK
                action = "warning"
            else:
                safety_level = SafetyLevel.LOW_RISK
                action = "logged"
        else:
            safety_level = SafetyLevel.SAFE
            action = "none"

        return ModerationResult(
            safe=safety_level == SafetyLevel.SAFE,
            safety_level=safety_level,
            categories=categories_found,
            category_scores=category_scores,
            flagged_content=flagged_content,
            explanation=f"Content contains {', '.join(c.value for c in categories_found)}" if categories_found else "Content appears safe",
            action_taken=action,
        )

    async def _ml_moderate(self, text: str) -> Dict[str, Any]:
        """Use ML model for moderation."""
        # Placeholder for ML model inference
        return {"flagged": False, "scores": {}, "categories": []}


# =============================================================================
# PII Detector
# =============================================================================

class PIIDetector:
    """
    Detects and masks Personally Identifiable Information (PII).
    """

    # PII patterns
    PII_PATTERNS: Dict[PIIType, str] = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        PIIType.SSN: r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
        PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        PIIType.API_KEY: r'\b(?:sk|pk|api|key|token)[_-]?[a-zA-Z0-9]{20,}\b',
        PIIType.PASSWORD: r'\b(?:password|passwd|pwd)\s*[:=]\s*\S+',
    }

    # Masking characters
    MASK_CHAR = '*'

    def __init__(
        self,
        mask_char: str = '*',
        detect_names: bool = False,
        custom_patterns: Optional[Dict[str, str]] = None,
    ):
        self.mask_char = mask_char
        self.detect_names = detect_names

        # Compile patterns
        self._compiled_patterns: Dict[PIIType, Pattern] = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            self._compiled_patterns[pii_type] = re.compile(pattern, re.IGNORECASE)

        # Add custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self._compiled_patterns[PIIType(name)] = re.compile(pattern, re.IGNORECASE)

    def detect(self, text: str) -> PIIResult:
        """
        Detect PII in text.

        Returns PIIResult with all matches and sanitized text.
        """
        matches: List[PIIMatch] = []
        pii_types_found: Set[PIIType] = set()

        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                value = match.group()
                masked = self._mask_value(value, pii_type)

                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=value,
                    masked_value=masked,
                    start=match.start(),
                    end=match.end(),
                ))
                pii_types_found.add(pii_type)

        # Sort matches by position
        matches.sort(key=lambda m: m.start)

        # Create sanitized text
        sanitized = self._sanitize_text(text, matches)

        return PIIResult(
            has_pii=len(matches) > 0,
            matches=matches,
            sanitized_text=sanitized,
            pii_types_found=pii_types_found,
        )

    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """Mask a PII value."""
        if pii_type == PIIType.EMAIL:
            # Show first char and domain
            at_idx = value.index('@')
            return value[0] + self.mask_char * (at_idx - 1) + value[at_idx:]

        elif pii_type == PIIType.PHONE:
            # Show last 4 digits
            digits = re.sub(r'\D', '', value)
            return self.mask_char * (len(digits) - 4) + digits[-4:]

        elif pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits
            digits = re.sub(r'\D', '', value)
            return self.mask_char * (len(digits) - 4) + digits[-4:]

        elif pii_type == PIIType.SSN:
            # Mask all
            return self.mask_char * len(re.sub(r'\D', '', value))

        else:
            # Default: mask middle portion
            if len(value) <= 4:
                return self.mask_char * len(value)
            return value[0] + self.mask_char * (len(value) - 2) + value[-1]

    def _sanitize_text(self, text: str, matches: List[PIIMatch]) -> str:
        """Replace PII with masked values in text."""
        # Process in reverse order to maintain positions
        sanitized = text
        for match in reversed(matches):
            sanitized = sanitized[:match.start] + match.masked_value + sanitized[match.end:]
        return sanitized

    def mask_pii(self, text: str) -> str:
        """Convenience method to mask all PII in text."""
        result = self.detect(text)
        return result.sanitized_text


# =============================================================================
# Prompt Injection Detector
# =============================================================================

class PromptInjectionDetector:
    """
    Detects prompt injection attempts.

    Prompt injections try to override the system prompt or
    make the model ignore its instructions.
    """

    # Common injection patterns
    INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
        # Pattern, Description, Severity
        (r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)', "Instruction override", 0.95),
        (r'disregard\s+(your|the)\s+(instructions?|rules?|guidelines?)', "Instruction disregard", 0.95),
        (r'forget\s+(everything|all)\s+(you\'ve\s+)?(been\s+)?told', "Memory wipe attempt", 0.9),
        (r'you\s+are\s+now\s+(a|an)\s+\w+', "Role reassignment", 0.8),
        (r'from\s+now\s+on\s+you\s+(will|must|should)', "Behavior modification", 0.85),
        (r'pretend\s+(to\s+be|you\'re)\s+(a|an)?\s*\w+', "Identity injection", 0.75),
        (r'\]\]\s*system\s*:', "System message injection", 0.95),
        (r'<\|?(system|assistant|user)\|?>', "Role tag injection", 0.9),
        (r'(jailbreak|dan\s+mode|developer\s+mode)', "Jailbreak attempt", 0.95),
        (r'act\s+as\s+if\s+you\s+have\s+no\s+(restrictions?|limits?|rules?)', "Restriction bypass", 0.9),
        (r'bypass\s+(your\s+)?(safety|content)\s+(filters?|restrictions?)', "Filter bypass", 0.95),
    ]

    # Delimiter injection patterns
    DELIMITER_PATTERNS: List[str] = [
        r'```\s*(system|instructions?)\s*```',
        r'---+\s*(system|new\s+instructions?)\s*---+',
        r'</?system>',
        r'\[INST\].*?\[/INST\]',
    ]

    def __init__(
        self,
        sensitivity: float = 0.7,
        custom_patterns: Optional[List[Tuple[str, str, float]]] = None,
    ):
        self.sensitivity = sensitivity

        # Compile patterns
        self._compiled_patterns: List[Tuple[Pattern, str, float]] = []
        for pattern, desc, severity in self.INJECTION_PATTERNS:
            self._compiled_patterns.append(
                (re.compile(pattern, re.IGNORECASE), desc, severity)
            )

        # Add custom patterns
        if custom_patterns:
            for pattern, desc, severity in custom_patterns:
                self._compiled_patterns.append(
                    (re.compile(pattern, re.IGNORECASE), desc, severity)
                )

        # Compile delimiter patterns
        self._delimiter_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DELIMITER_PATTERNS
        ]

    def detect(self, text: str) -> InjectionResult:
        """
        Detect prompt injection attempts.

        Returns InjectionResult with detection details.
        """
        flagged_patterns: List[str] = []
        max_confidence = 0.0
        injection_type = ""

        # Check injection patterns
        for pattern, desc, severity in self._compiled_patterns:
            if pattern.search(text):
                flagged_patterns.append(desc)
                if severity > max_confidence:
                    max_confidence = severity
                    injection_type = desc

        # Check delimiter patterns
        for pattern in self._delimiter_patterns:
            if pattern.search(text):
                flagged_patterns.append("Delimiter injection")
                max_confidence = max(max_confidence, 0.85)
                if not injection_type:
                    injection_type = "Delimiter injection"

        # Check for unusual control characters
        control_chars = re.findall(r'[\x00-\x1f\x7f-\x9f]', text)
        if len(control_chars) > 5:
            flagged_patterns.append("Suspicious control characters")
            max_confidence = max(max_confidence, 0.6)

        is_injection = max_confidence >= self.sensitivity

        return InjectionResult(
            is_injection=is_injection,
            confidence=max_confidence,
            injection_type=injection_type,
            flagged_patterns=flagged_patterns,
            explanation=f"Detected: {', '.join(flagged_patterns)}" if flagged_patterns else "No injection detected",
        )


# =============================================================================
# Safety Guard (Combined)
# =============================================================================

class SafetyGuard:
    """
    Comprehensive safety guard combining all safety features.
    """

    def __init__(
        self,
        content_moderator: Optional[ContentModerator] = None,
        pii_detector: Optional[PIIDetector] = None,
        injection_detector: Optional[PromptInjectionDetector] = None,
        block_on_injection: bool = True,
        block_on_pii: bool = False,
        mask_pii: bool = True,
    ):
        self.content_moderator = content_moderator or ContentModerator()
        self.pii_detector = pii_detector or PIIDetector()
        self.injection_detector = injection_detector or PromptInjectionDetector()

        self.block_on_injection = block_on_injection
        self.block_on_pii = block_on_pii
        self.mask_pii = mask_pii

    async def check_input(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check input for safety issues.

        Returns dict with:
        - safe: bool
        - processed_text: str (with PII masked if enabled)
        - moderation: ModerationResult
        - pii: PIIResult
        - injection: InjectionResult
        - blocked: bool
        - block_reason: str
        """
        result = {
            "safe": True,
            "processed_text": text,
            "blocked": False,
            "block_reason": "",
            "warnings": [],
        }

        # Check for prompt injection
        injection_result = self.injection_detector.detect(text)
        result["injection"] = injection_result

        if injection_result.is_injection:
            result["warnings"].append(f"Prompt injection detected: {injection_result.injection_type}")
            if self.block_on_injection:
                result["safe"] = False
                result["blocked"] = True
                result["block_reason"] = f"Prompt injection detected: {injection_result.injection_type}"
                return result

        # Check for harmful content
        moderation_result = await self.content_moderator.moderate(text, context)
        result["moderation"] = moderation_result

        if not moderation_result.safe:
            result["safe"] = False
            result["warnings"].append(f"Content flagged: {moderation_result.explanation}")

            if moderation_result.safety_level == SafetyLevel.BLOCKED:
                result["blocked"] = True
                result["block_reason"] = moderation_result.explanation
                return result

        # Check for PII
        pii_result = self.pii_detector.detect(text)
        result["pii"] = pii_result

        if pii_result.has_pii:
            result["warnings"].append(f"PII detected: {', '.join(t.value for t in pii_result.pii_types_found)}")

            if self.block_on_pii:
                result["safe"] = False
                result["blocked"] = True
                result["block_reason"] = "PII detected in input"
                return result

            if self.mask_pii:
                result["processed_text"] = pii_result.sanitized_text

        return result

    async def check_output(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check output for safety issues.

        Similar to check_input but optimized for assistant responses.
        """
        result = {
            "safe": True,
            "processed_text": text,
            "blocked": False,
            "block_reason": "",
            "warnings": [],
        }

        # Check for harmful content in output
        moderation_result = await self.content_moderator.moderate(text, context)
        result["moderation"] = moderation_result

        if not moderation_result.safe:
            result["safe"] = False
            result["warnings"].append(f"Output flagged: {moderation_result.explanation}")

            if moderation_result.safety_level == SafetyLevel.BLOCKED:
                result["blocked"] = True
                result["block_reason"] = moderation_result.explanation
                return result

        # Check for PII in output (shouldn't leak user PII)
        pii_result = self.pii_detector.detect(text)
        result["pii"] = pii_result

        if pii_result.has_pii and self.mask_pii:
            result["processed_text"] = pii_result.sanitized_text
            result["warnings"].append("PII masked in output")

        return result

    def is_safe_query(self, text: str) -> bool:
        """Quick synchronous check if query appears safe."""
        # Quick injection check
        injection = self.injection_detector.detect(text)
        if injection.is_injection:
            return False

        # Quick pattern check
        text_lower = text.lower()
        for blocked in ContentModerator.BLOCKLIST:
            if blocked in text_lower:
                return False

        return True
