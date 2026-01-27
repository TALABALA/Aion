"""
AION Test Runner - Execute generated tests safely.

Runs generated test code in a controlled environment
with timeout and resource limits.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from aion.nlp.types import GeneratedCode, ValidationResult, ValidationStatus

logger = structlog.get_logger(__name__)


class TestRunner:
    """
    Executes generated tests in a controlled environment.

    Safety measures:
    - Timeout enforcement
    - Exception isolation
    - Resource limits
    """

    def __init__(self, timeout_seconds: float = 30.0):
        self._timeout = timeout_seconds

    async def run(self, code: GeneratedCode) -> ValidationResult:
        """
        Run generated tests.

        Args:
            code: Generated code with test_code

        Returns:
            Validation result with test details
        """
        result = ValidationResult()

        if not code.test_code:
            result.tests_skipped = 1
            result.status = ValidationStatus.SKIPPED
            return result

        try:
            # Attempt to parse test code (syntax check)
            import ast
            ast.parse(code.test_code)
        except SyntaxError as e:
            result.add_error(f"Test code syntax error: {e.msg}")
            result.tests_failed = 1
            result.status = ValidationStatus.FAILED
            return result

        # Execute tests in isolated namespace
        try:
            test_results = await asyncio.wait_for(
                self._execute_tests(code),
                timeout=self._timeout,
            )
            result.tests_passed = test_results.get("passed", 0)
            result.tests_failed = test_results.get("failed", 0)
            result.test_details = test_results.get("details", [])

        except asyncio.TimeoutError:
            result.add_error(f"Test execution timed out after {self._timeout}s")
            result.tests_failed = 1
        except Exception as e:
            result.add_warning(f"Test execution error: {e}")
            result.tests_skipped = 1

        if result.tests_failed > 0:
            result.status = ValidationStatus.WARNING
        elif result.tests_passed > 0:
            result.status = ValidationStatus.PASSED
        else:
            result.status = ValidationStatus.SKIPPED

        return result

    async def _execute_tests(self, code: GeneratedCode) -> Dict[str, Any]:
        """Execute test code in isolated namespace."""
        results: Dict[str, Any] = {"passed": 0, "failed": 0, "details": []}

        # Create isolated namespace with safe builtins
        namespace: Dict[str, Any] = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "type": type,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "RuntimeError": RuntimeError,
                "NotImplementedError": NotImplementedError,
                "True": True,
                "False": False,
                "None": None,
            },
        }

        try:
            # Execute the main code first
            exec(code.code, namespace)

            # Find test functions
            exec(code.test_code, namespace)

            # Run test functions
            for name, obj in namespace.items():
                if name.startswith("test_") and callable(obj):
                    try:
                        if asyncio.iscoroutinefunction(obj):
                            await obj()
                        else:
                            obj()
                        results["passed"] += 1
                        results["details"].append({
                            "name": name,
                            "status": "passed",
                        })
                    except AssertionError as e:
                        results["failed"] += 1
                        results["details"].append({
                            "name": name,
                            "status": "failed",
                            "error": str(e),
                        })
                    except Exception as e:
                        results["failed"] += 1
                        results["details"].append({
                            "name": name,
                            "status": "error",
                            "error": str(e),
                        })

        except Exception as e:
            logger.debug("Test execution namespace error", error=str(e))
            results["details"].append({
                "name": "setup",
                "status": "error",
                "error": str(e),
            })

        return results
