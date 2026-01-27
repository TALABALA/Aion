"""
AION Natural Language Programming - Configuration

Pydantic-based configuration following AION conventions for
type-safe, environment-driven settings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IntentClassificationConfig(BaseModel):
    """Configuration for the intent classification pipeline."""

    # Ensemble weights: pattern (fast, deterministic) + LLM (deep, semantic)
    # Weights are normalized at scoring time, so they don't need to sum to 1.0
    llm_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    pattern_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    # Confidence thresholds
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    clarification_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Multi-intent detection
    enable_multi_intent: bool = True
    max_sub_intents: int = 5


class EntityExtractionConfig(BaseModel):
    """Configuration for entity extraction."""

    # Extraction methods
    use_regex: bool = True
    use_llm: bool = True

    # Confidence
    min_entity_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Limits
    max_entities_per_type: int = 20


class SynthesisConfig(BaseModel):
    """Configuration for code synthesis."""

    # Generation
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_generation_tokens: int = 4096
    generation_model: Optional[str] = None

    # Code style
    target_language: str = "python"
    code_style: Literal["functional", "oop", "mixed"] = "mixed"
    include_type_hints: bool = True
    include_docstrings: bool = True
    include_error_handling: bool = True

    # Testing
    generate_tests: bool = True
    min_test_coverage: float = Field(default=0.8, ge=0.0, le=1.0)

    # Max retries for generation
    max_generation_retries: int = 3


class ValidationConfig(BaseModel):
    """Configuration for code validation."""

    # Checks to run
    enable_syntax_check: bool = True
    enable_static_analysis: bool = True
    enable_safety_check: bool = True
    enable_test_execution: bool = True
    enable_complexity_check: bool = True

    # Safety
    max_safety_risk: Literal["safe", "low_risk", "medium_risk"] = "low_risk"
    blocked_imports: List[str] = Field(default_factory=lambda: [
        "os.system",
        "subprocess",
        "shutil.rmtree",
        "ctypes",
        "importlib",
    ])
    blocked_builtins: List[str] = Field(default_factory=lambda: [
        "eval",
        "exec",
        "__import__",
        "compile",
        "globals",
        "locals",
    ])

    # Complexity thresholds
    max_cyclomatic_complexity: int = 15
    max_function_lines: int = 100
    max_nesting_depth: int = 5

    # Test execution
    test_timeout_seconds: float = 30.0
    sandbox_execution: bool = True


class DeploymentConfig(BaseModel):
    """Configuration for deployment."""

    # Deployment behavior
    require_validation: bool = True
    require_confirmation: bool = True
    auto_deploy_on_pass: bool = False

    # Versioning
    enable_versioning: bool = True
    max_versions: int = 50

    # Rollback
    enable_rollback: bool = True
    auto_rollback_on_error: bool = True
    error_rate_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    # Registry (memory-backed; persistent backends planned for future)
    registry_backend: Literal["memory"] = "memory"


class SessionConfig(BaseModel):
    """Configuration for programming sessions."""

    # Session limits
    max_iterations: int = 50
    max_session_duration_seconds: float = 3600.0
    max_messages: int = 200

    # Context
    context_window_messages: int = 20
    enable_suggestions: bool = True
    max_suggestions: int = 5

    # Cleanup
    idle_timeout_seconds: float = 1800.0
    auto_cleanup: bool = True


class NLProgrammingConfig(BaseModel):
    """
    Main configuration for the Natural Language Programming system.

    All sub-configurations use sensible defaults and can be overridden
    via environment variables with the AION_NLP_ prefix.
    """

    # Feature flags
    enabled: bool = True

    # Sub-configurations
    intent: IntentClassificationConfig = Field(
        default_factory=IntentClassificationConfig
    )
    entities: EntityExtractionConfig = Field(
        default_factory=EntityExtractionConfig
    )
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    # Logging
    log_intents: bool = True
    log_generated_code: bool = False
    log_validation_details: bool = True

    # Performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 256
    max_concurrent_sessions: int = 100

    # Circuit breaker for LLM calls
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 30.0
