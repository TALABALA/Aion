"""
AION Safety Analyzer - Security analysis for generated code.

SOTA safety analysis using multi-layer threat detection:
- Pattern-based detection for known dangerous constructs
- AST-based import and call analysis
- Stub/incomplete code detection
- Graduated scoring with aggressive thresholds for critical patterns
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


def score_to_safety_level(score: float) -> SafetyLevel:
    """Convert a safety score [0.0, 1.0] to a SafetyLevel.

    This is the single source of truth for safety threshold boundaries.
    Used by both SafetyAnalyzer and ValidationResult.merge().
    """
    if score >= 0.9:
        return SafetyLevel.SAFE
    elif score >= 0.7:
        return SafetyLevel.LOW_RISK
    elif score >= 0.5:
        return SafetyLevel.MEDIUM_RISK
    elif score >= 0.25:
        return SafetyLevel.HIGH_RISK
    else:
        return SafetyLevel.DANGEROUS


# Critical patterns: any single match immediately fails validation
_CRITICAL_PATTERNS: List[Dict[str, Any]] = [
    {"pattern": r"\bos\.system\s*\(", "name": "os.system()", "reason": "Arbitrary shell command execution"},
    {"pattern": r"\beval\s*\(", "name": "eval()", "reason": "Arbitrary code execution via eval"},
    {"pattern": r"\bexec\s*\(", "name": "exec()", "reason": "Arbitrary code execution via exec"},
    {"pattern": r"\b__import__\s*\(", "name": "__import__()", "reason": "Dynamic import bypasses static analysis"},
    {"pattern": r"\bshutil\.rmtree\b", "name": "shutil.rmtree()", "reason": "Recursive directory deletion"},
]

# High-risk patterns: significantly reduce safety score
_HIGH_RISK_PATTERNS: List[Dict[str, Any]] = [
    {"pattern": r"\bsubprocess\.", "name": "subprocess", "weight": 0.4, "reason": "Command execution via subprocess"},
    {"pattern": r"\bcompile\s*\(", "name": "compile()", "weight": 0.35, "reason": "Dynamic code compilation"},
    {"pattern": r"\bctypes\b", "name": "ctypes", "weight": 0.4, "reason": "Low-level memory access"},
    {"pattern": r"\bpickle\.loads?\b", "name": "pickle", "weight": 0.3, "reason": "Unsafe deserialization (arbitrary code execution)"},
    {"pattern": r"\byaml\.(?:load|unsafe_load)\b", "name": "yaml.load", "weight": 0.25, "reason": "Unsafe YAML loading"},
    {"pattern": r"\bglobals\s*\(\s*\)", "name": "globals()", "weight": 0.35, "reason": "Global namespace manipulation"},
    {"pattern": r"\bos\.remove\b", "name": "os.remove()", "weight": 0.2, "reason": "File deletion"},
    {"pattern": r"\bos\.rmdir\b", "name": "os.rmdir()", "weight": 0.2, "reason": "Directory deletion"},
    {"pattern": r"\bsocket\b", "name": "socket", "weight": 0.15, "reason": "Raw network socket access"},
    {"pattern": r"\bwebbrowser\.", "name": "webbrowser", "weight": 0.1, "reason": "Browser control"},
]

# Stub / incomplete code patterns that should not pass validation
_STUB_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"\braise\s+NotImplementedError\b", "name": "NotImplementedError", "reason": "Contains unimplemented stub code"},
    {"pattern": r"#\s*TODO\s*:", "name": "TODO comment", "reason": "Contains TODO markers indicating incomplete implementation"},
    {"pattern": r"#\s*FIXME\s*:", "name": "FIXME comment", "reason": "Contains FIXME markers indicating known issues"},
    {"pattern": r"\bpass\s*#", "name": "pass with comment", "reason": "Contains empty pass statements (likely stub code)"},
]

# Safe import allowlist
_SAFE_IMPORTS: Set[str] = {
    "json", "re", "datetime", "typing",
    "asyncio", "dataclasses", "enum", "uuid",
    "math", "statistics", "collections", "itertools",
    "functools", "operator", "string", "textwrap",
    "hashlib", "hmac", "base64",
    "logging", "structlog",
    "pydantic", "fastapi",
    "aiohttp", "httpx",
    "aion",
}

# Imports that are conditionally safe (need context)
_CONDITIONAL_IMPORTS: Dict[str, str] = {
    "pathlib": "Filesystem access (path traversal risk if user-controlled)",
    "urllib": "Network requests (SSRF risk if URL is user-controlled)",
    "tempfile": "Temporary file creation",
}


class SafetyAnalyzer:
    """
    Analyzes generated code for security concerns using multi-layer detection.

    Layer 1: Critical pattern detection (immediate fail)
    Layer 2: High-risk pattern scoring (graduated)
    Layer 3: AST-based import analysis
    Layer 4: Stub/incomplete code detection
    Layer 5: File and network operation checks
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

        # Layer 1: Critical patterns (immediate DANGEROUS)
        critical_found = self._check_critical_patterns(code, result)
        if critical_found:
            result.safety_score = 0.0
            result.safety_level = SafetyLevel.DANGEROUS
            result.add_error("Code contains critical security patterns and cannot be deployed")
            result.status = ValidationStatus.FAILED
            return result

        # Layer 1b: Config-driven blocked builtins check
        blocked_found = self._check_blocked_builtins(code, result)
        if blocked_found:
            result.safety_score = 0.0
            result.safety_level = SafetyLevel.DANGEROUS
            result.add_error("Code uses blocked builtin functions")
            result.status = ValidationStatus.FAILED
            return result

        # Layer 2: High-risk patterns (graduated scoring)
        risk_score = self._check_high_risk_patterns(code, result)

        # Layer 3: Import analysis
        import_risk = self._analyze_imports(code, result)
        risk_score += import_risk

        # Layer 4: Stub detection (capped to prevent over-penalization)
        stub_count = self._check_stubs(code, result)
        if stub_count > 0:
            risk_score += min(0.2 * stub_count, 0.4)

        # Layer 5: File and network operations
        risk_score += self._check_io_operations(code, result)

        # Calculate final safety score
        # Score formula: 1.0 - risk, clamped to [0.0, 1.0]
        result.safety_score = max(0.0, min(1.0, 1.0 - risk_score))
        result.safety_level = self._score_to_level(result.safety_score)

        if result.safety_level == SafetyLevel.DANGEROUS:
            result.add_error("Code safety score too low for deployment")
            result.status = ValidationStatus.FAILED
        elif result.safety_level == SafetyLevel.HIGH_RISK:
            result.add_warning("Code has high-risk patterns - review before deployment")
            result.status = ValidationStatus.WARNING
        elif result.safety_level == SafetyLevel.MEDIUM_RISK:
            result.status = ValidationStatus.WARNING
        else:
            result.status = ValidationStatus.PASSED

        return result

    def _check_critical_patterns(self, code: str, result: ValidationResult) -> bool:
        """Check for critical patterns that immediately fail validation."""
        found = False
        for pattern_def in _CRITICAL_PATTERNS:
            if re.search(pattern_def["pattern"], code):
                found = True
                result.safety_concerns.append(
                    f"CRITICAL: {pattern_def['name']} - {pattern_def['reason']}"
                )
                result.add_error(
                    f"Critical security pattern: {pattern_def['name']} - {pattern_def['reason']}",
                    rule="critical_safety",
                )
        return found

    def _check_high_risk_patterns(self, code: str, result: ValidationResult) -> float:
        """Check for high-risk patterns, return cumulative risk score."""
        total_risk = 0.0
        for pattern_def in _HIGH_RISK_PATTERNS:
            if re.search(pattern_def["pattern"], code):
                weight = pattern_def["weight"]
                total_risk += weight
                result.safety_concerns.append(
                    f"{pattern_def['name']}: {pattern_def['reason']}"
                )
                result.add_warning(
                    f"High-risk pattern: {pattern_def['name']} - {pattern_def['reason']}",
                    rule="safety_pattern",
                )
        return total_risk

    def _check_stubs(self, code: str, result: ValidationResult) -> int:
        """Detect stub/incomplete code. Returns count of stubs found."""
        count = 0
        for stub_def in _STUB_PATTERNS:
            matches = re.findall(stub_def["pattern"], code, re.IGNORECASE)
            if matches:
                count += len(matches)
                result.safety_concerns.append(
                    f"Incomplete code: {stub_def['name']} ({len(matches)} occurrence(s))"
                )
                result.add_warning(
                    f"Incomplete code detected: {stub_def['reason']} ({len(matches)} found)",
                    rule="stub_detection",
                )
        return count

    def _check_io_operations(self, code: str, result: ValidationResult) -> float:
        """Check file and network I/O operations. Returns risk score."""
        risk = 0.0

        # File write operations
        if re.search(r"\bopen\s*\(", code):
            if re.search(r"\bopen\s*\(.+,\s*['\"]w", code):
                risk += 0.15
                result.add_warning(
                    "File write operation detected - ensure path is validated",
                    rule="file_write",
                    suggestion="Use pathlib for safe path construction and validate against path traversal",
                )

        # Network access without error handling
        if re.search(r"\baiohttp\b|\brequests\b|\bhttpx\b", code):
            if not re.search(r"\bresponse\.raise_for_status\b", code):
                risk += 0.05
                result.add_warning(
                    "HTTP request without status check",
                    rule="unchecked_http",
                    suggestion="Add response.raise_for_status() or explicit status code handling",
                )

        return risk

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
                    risk += self._score_import(module, alias.name, node.lineno, result)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    risk += self._score_import(module, node.module, node.lineno, result)

        return risk

    def _score_import(
        self,
        module: str,
        full_name: str,
        lineno: int,
        result: ValidationResult,
    ) -> float:
        """Score a single import for risk. Returns risk contribution."""
        # Blocked imports
        if full_name in self._blocked_imports or module in self._blocked_imports:
            result.add_error(
                f"Blocked import: {full_name}",
                rule="blocked_import",
                line=lineno,
            )
            return 0.5

        # Safe imports
        if module in _SAFE_IMPORTS:
            return 0.0

        # Conditionally safe imports
        if module in _CONDITIONAL_IMPORTS:
            result.add_warning(
                f"Conditional import: {full_name} - {_CONDITIONAL_IMPORTS[module]}",
                rule="conditional_import",
                line=lineno,
            )
            return 0.05

        # Unknown imports
        result.add_warning(
            f"Non-allowlisted import: {full_name}",
            rule="import_check",
            line=lineno,
        )
        return 0.1

    def _check_blocked_builtins(self, code: str, result: ValidationResult) -> bool:
        """Check for config-driven blocked builtins."""
        if not self._blocked_builtins:
            return False

        found = False
        for name in self._blocked_builtins:
            pattern = rf"\b{re.escape(name)}\s*\("
            if re.search(pattern, code):
                found = True
                result.safety_concerns.append(
                    f"BLOCKED BUILTIN: {name}() - configured as blocked"
                )
                result.add_error(
                    f"Blocked builtin: {name}()",
                    rule="blocked_builtin",
                )
        return found

    @staticmethod
    def _score_to_level(score: float) -> SafetyLevel:
        """Convert safety score to level using aggressive thresholds."""
        return score_to_safety_level(score)
