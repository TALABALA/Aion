"""
AION Validation Engine - Multi-stage code validation pipeline.

Orchestrates syntax checking, static analysis, safety analysis,
and test execution for generated code.
"""

from __future__ import annotations

import time
from typing import Any, Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import (
    GeneratedCode,
    SafetyLevel,
    ValidationResult,
    ValidationStatus,
)
from aion.nlp.validation.syntax import SyntaxChecker
from aion.nlp.validation.safety import SafetyAnalyzer
from aion.nlp.validation.testing import TestRunner

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel
    from aion.nlp.config import ValidationConfig

logger = structlog.get_logger(__name__)


class ValidationEngine:
    """
    Multi-stage validation pipeline for generated code.

    Pipeline stages:
    1. Syntax validation (AST parsing)
    2. Static analysis (unused vars, complexity)
    3. Safety analysis (dangerous patterns, imports)
    4. Test execution (in sandboxed environment)
    """

    def __init__(self, kernel: AIONKernel, config: Optional[ValidationConfig] = None):
        self.kernel = kernel
        self._config = config

        self._syntax = SyntaxChecker()
        self._safety = SafetyAnalyzer(config=config)
        self._tests = TestRunner()

    async def validate(self, code: GeneratedCode) -> ValidationResult:
        """
        Run the full validation pipeline.

        Args:
            code: Generated code to validate

        Returns:
            Comprehensive validation result
        """
        start = time.monotonic()
        result = ValidationResult()

        # Stage 1: Syntax
        if self._should_run("syntax"):
            syntax_result = self._syntax.check(code.code)
            result.merge(syntax_result)
            if syntax_result.errors:
                result.status = ValidationStatus.FAILED
                result.validation_time_ms = (time.monotonic() - start) * 1000
                logger.info("Validation failed at syntax stage", errors=len(syntax_result.errors))
                return result

        # Stage 2: Safety
        if self._should_run("safety"):
            safety_result = self._safety.analyze(code.code)
            result.merge(safety_result)
            if safety_result.safety_level == SafetyLevel.DANGEROUS:
                result.status = ValidationStatus.FAILED
                result.validation_time_ms = (time.monotonic() - start) * 1000
                logger.warning("Validation failed: dangerous code detected")
                return result

        # Stage 3: Static analysis (part of syntax checker)
        if self._should_run("static"):
            static_result = self._syntax.static_analyze(code.code)
            result.merge(static_result)

        # Stage 4: Tests
        if self._should_run("tests") and code.test_code:
            test_result = await self._tests.run(code)
            result.merge(test_result)

        # Determine final status
        if result.errors:
            result.status = ValidationStatus.FAILED
        elif result.warnings:
            result.status = ValidationStatus.WARNING
        else:
            result.status = ValidationStatus.PASSED

        result.validation_time_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Validation complete",
            status=result.status.value,
            errors=len(result.errors),
            warnings=len(result.warnings),
            safety=result.safety_level.value,
            time_ms=round(result.validation_time_ms, 1),
        )

        return result

    async def quick_validate(self, code: str) -> ValidationResult:
        """Quick validation (syntax + safety only)."""
        result = ValidationResult()

        syntax_result = self._syntax.check(code)
        result.merge(syntax_result)

        if not syntax_result.errors:
            safety_result = self._safety.analyze(code)
            result.merge(safety_result)

        if result.errors:
            result.status = ValidationStatus.FAILED
        elif result.warnings:
            result.status = ValidationStatus.WARNING
        else:
            result.status = ValidationStatus.PASSED

        return result

    def _should_run(self, stage: str) -> bool:
        """Check if a validation stage should run."""
        if not self._config:
            return True
        return {
            "syntax": self._config.enable_syntax_check,
            "safety": self._config.enable_safety_check,
            "static": self._config.enable_static_analysis,
            "tests": self._config.enable_test_execution,
        }.get(stage, True)
