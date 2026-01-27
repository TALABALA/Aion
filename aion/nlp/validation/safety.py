"""
AION Safety Analyzer - Security analysis for generated code.

Detects dangerous patterns, unsafe imports, and potential
security vulnerabilities in generated code.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import structlog

from aion.nlp.types import (
    SafetyLevel,
    ValidationIssue,
    ValidationResult,
    ValidationStatus,
)

if TYPE_CHECKING:
    from aion.nlp.config import ValidationConfig

logger = structlog.get_logger(__name__)


# Dangerous patterns with severity weights
_DANGEROUS_PATTERNS: List[Dict[str, Any]] = [
    {"pattern": r"\bos\.system\s*\(", "name": "os.system", "weight": 1.0, "reason": "Arbitrary command execution"},
    {"pattern": r"\bsubprocess\.", "name": "subprocess", "weight": 0.8, "reason": "Command execution"},
    {"pattern": r"\beval\s*\(", "name": "eval()", "weight": 1.0, "reason": "Arbitrary code execution"},
    {"pattern": r"\bexec\s*\(", "name": "exec()", "weight": 1.0, "reason": "Arbitrary code execution"},
    {"pattern": r"\b__import__\s*\(", "name": "__import__()", "weight": 0.9, "reason": "Dynamic import"},
    {"pattern": r"\bcompile\s*\(", "name": "compile()", "weight": 0.7, "reason": "Dynamic code compilation"},
    {"pattern": r"\bshutil\.rmtree\b", "name": "shutil.rmtree", "weight": 0.9, "reason": "Recursive directory deletion"},
    {"pattern": r"\bos\.remove\b", "name": "os.remove", "weight": 0.5, "reason": "File deletion"},
    {"pattern": r"\bos\.rmdir\b", "name": "os.rmdir", "weight": 0.5, "reason": "Directory deletion"},
    {"pattern": r"\bctypes\b", "name": "ctypes", "weight": 0.8, "reason": "Low-level memory access"},
    {"pattern": r"\bpickle\.loads?\b", "name": "pickle", "weight": 0.6, "reason": "Unsafe deserialization"},
    {"pattern": r"\byaml\.(?:load|unsafe_load)\b", "name": "yaml.load", "weight": 0.5, "reason": "Unsafe YAML loading"},
    {"pattern": r"\bglobals\s*\(\s*\)", "name": "globals()", "weight": 0.7, "reason": "Global namespace access"},
    {"pattern": r"\bsocket\b", "name": "socket", "weight": 0.4, "reason": "Network socket access"},
    {"pattern": r"\bwebbrowser\.", "name": "webbrowser", "weight": 0.3, "reason": "Browser control"},
]

# Safe import allowlist
_SAFE_IMPORTS: Set[str] = {
    "aiohttp", "json", "re", "datetime", "typing",
    "asyncio", "dataclasses", "enum", "uuid",
    "math", "statistics", "collections", "itertools",
    "functools", "operator", "string", "textwrap",
    "hashlib", "hmac", "base64", "urllib",
    "logging", "structlog", "pathlib",
    "pydantic", "fastapi",
    "aion",
}


class SafetyAnalyzer:
    """
    Analyzes generated code for security concerns.

    Uses a combination of:
    - Pattern matching for known dangerous constructs
    - AST analysis for import verification
    - Scoring system for risk assessment
    """

    def __init__(self, config: Optional[Any] = None):
        self._config = config
        self._blocked_imports: Set[str] = set()
        self._blocked_builtins: Set[str] = set()

        if config:
            self._blocked_imports = set(getattr(config, 'blocked_imports', []))
            self._blocked_builtins = set(getattr(config, 'blocked_builtins', []))

    def analyze(self, code: str) -> ValidationResult:
        """
        Analyze code for safety concerns.

        Returns a ValidationResult with safety scoring.
        """
        result = ValidationResult()
        total_risk = 0.0

        # Pattern-based detection
        for pattern_def in _DANGEROUS_PATTERNS:
            if re.search(pattern_def["pattern"], code):
                weight = pattern_def["weight"]
                total_risk += weight
                result.safety_concerns.append(
                    f"{pattern_def['name']}: {pattern_def['reason']}"
                )
                if weight >= 0.8:
                    result.add_warning(
                        f"High-risk pattern detected: {pattern_def['name']} - {pattern_def['reason']}",
                        rule="safety_pattern",
                    )
                else:
                    result.add_warning(
                        f"Potentially unsafe pattern: {pattern_def['name']}",
                        rule="safety_pattern",
                    )

        # Import analysis
        import_risk = self._analyze_imports(code, result)
        total_risk += import_risk

        # Open file operations check
        if re.search(r"\bopen\s*\(", code):
            if re.search(r"\bopen\s*\(.+,\s*['\"]w['\"]", code):
                total_risk += 0.3
                result.add_warning(
                    "File write operation detected - ensure path is validated",
                    rule="file_write",
                )

        # Network access check
        if re.search(r"\baiohttp\b|\brequests\b|\bhttpx\b", code):
            if not re.search(r"\bresponse\.raise_for_status\b", code):
                result.add_warning(
                    "HTTP request without status check - consider adding error handling",
                    rule="unchecked_http",
                    suggestion="Add response.raise_for_status() after HTTP requests",
                )

        # Calculate safety score and level
        result.safety_score = max(0.0, 1.0 - total_risk * 0.3)
        result.safety_level = self._score_to_level(result.safety_score)

        if result.safety_level == SafetyLevel.DANGEROUS:
            result.add_error("Code contains dangerous patterns and cannot be deployed")
            result.status = ValidationStatus.FAILED
        elif result.safety_level == SafetyLevel.HIGH_RISK:
            result.status = ValidationStatus.WARNING
        else:
            result.status = ValidationStatus.PASSED

        return result

    def _analyze_imports(self, code: str, result: ValidationResult) -> float:
        """Analyze imports for safety. Returns risk score."""
        risk = 0.0

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in _SAFE_IMPORTS:
                        risk += 0.2
                        result.add_warning(
                            f"Non-standard import: {alias.name}",
                            rule="import_check",
                            line=node.lineno,
                        )
                    if alias.name in self._blocked_imports:
                        risk += 0.5
                        result.add_error(
                            f"Blocked import: {alias.name}",
                            rule="blocked_import",
                            line=node.lineno,
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module not in _SAFE_IMPORTS:
                        risk += 0.2
                        result.add_warning(
                            f"Non-standard import: from {node.module}",
                            rule="import_check",
                            line=node.lineno,
                        )

        return risk

    def _score_to_level(self, score: float) -> SafetyLevel:
        """Convert safety score to level."""
        if score >= 0.9:
            return SafetyLevel.SAFE
        elif score >= 0.7:
            return SafetyLevel.LOW_RISK
        elif score >= 0.5:
            return SafetyLevel.MEDIUM_RISK
        elif score >= 0.3:
            return SafetyLevel.HIGH_RISK
        else:
            return SafetyLevel.DANGEROUS
