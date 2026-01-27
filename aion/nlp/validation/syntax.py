"""
AION Syntax Checker - AST-based syntax and static analysis.

Validates Python code syntax and performs basic static analysis
including complexity checking and pattern detection.
"""

from __future__ import annotations

import ast
from typing import Any, List, Optional

import structlog

from aion.nlp.types import ValidationIssue, ValidationResult, ValidationStatus

logger = structlog.get_logger(__name__)


class SyntaxChecker:
    """
    AST-based Python syntax validation and static analysis.

    Checks:
    - Syntax correctness
    - Basic code quality patterns
    - Complexity metrics
    - Common anti-patterns
    """

    def check(self, code: str) -> ValidationResult:
        """Check Python syntax."""
        result = ValidationResult()

        try:
            ast.parse(code)
            result.status = ValidationStatus.PASSED
        except SyntaxError as e:
            result.add_error(
                f"Syntax error at line {e.lineno}: {e.msg}",
                line=e.lineno,
                column=e.offset,
                rule="syntax",
            )
            result.status = ValidationStatus.FAILED

        return result

    def static_analyze(self, code: str) -> ValidationResult:
        """Perform static analysis on Python code."""
        result = ValidationResult()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Syntax errors handled by check()
            return result

        analyzer = _StaticAnalyzer()
        analyzer.visit(tree)

        for issue in analyzer.issues:
            result.issues.append(issue)

        if any(i.is_error for i in result.issues):
            result.status = ValidationStatus.FAILED
        elif result.issues:
            result.status = ValidationStatus.WARNING
        else:
            result.status = ValidationStatus.PASSED

        return result


class _StaticAnalyzer(ast.NodeVisitor):
    """AST visitor for static code analysis."""

    def __init__(self) -> None:
        self.issues: List[ValidationIssue] = []
        self._function_depth = 0
        self._nesting_depth = 0
        self._max_nesting = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_depth += 1
        self._check_function(node)
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_depth += 1
        self._check_function(node)
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.issues.append(ValidationIssue(
                severity="warning",
                message="Bare except clause - consider catching specific exceptions",
                line=node.lineno,
                rule="bare_except",
                suggestion="Use 'except Exception as e:' instead of bare 'except:'",
            ))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._nesting_depth += 1
        self._max_nesting = max(self._max_nesting, self._nesting_depth)
        if self._nesting_depth > 5:
            self.issues.append(ValidationIssue(
                severity="warning",
                message=f"Deep nesting (depth={self._nesting_depth}): consider refactoring",
                line=node.lineno,
                rule="deep_nesting",
                suggestion="Extract nested logic into helper functions",
            ))
        self.generic_visit(node)
        self._nesting_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._nesting_depth += 1
        self._max_nesting = max(self._max_nesting, self._nesting_depth)
        self.generic_visit(node)
        self._nesting_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._nesting_depth += 1
        self._max_nesting = max(self._max_nesting, self._nesting_depth)
        self.generic_visit(node)
        self._nesting_depth -= 1

    def _check_function(self, node: Any) -> None:
        """Check function-level issues."""
        # Check function length
        if hasattr(node, 'body'):
            line_count = 0
            for child in ast.walk(node):
                if hasattr(child, 'lineno'):
                    line_count += 1
            if line_count > 100:
                self.issues.append(ValidationIssue(
                    severity="warning",
                    message=f"Function '{node.name}' is very long ({line_count} AST nodes)",
                    line=node.lineno,
                    rule="function_length",
                    suggestion="Consider breaking this function into smaller functions",
                ))

        # Check parameter count
        args = node.args
        param_count = len(args.args) + len(args.kwonlyargs)
        if param_count > 10:
            self.issues.append(ValidationIssue(
                severity="warning",
                message=f"Function '{node.name}' has {param_count} parameters",
                line=node.lineno,
                rule="too_many_params",
                suggestion="Consider using a configuration object or dataclass",
            ))
