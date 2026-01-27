"""
AION Sandbox Executor - Safe code execution environment.

Provides sandboxed execution with restricted builtins,
resource limits, and isolation for testing generated code.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


# Restricted builtins for sandbox
_SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "bytes", "chr",
    "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hash",
    "hex", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min",
    "next", "oct", "ord", "pow", "print",
    "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip",
    # Exceptions
    "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "RuntimeError",
    "NotImplementedError", "StopIteration",
    "AttributeError", "ImportError",
    # Constants
    "True", "False", "None",
}


class SandboxExecutor:
    """
    Sandboxed code execution environment.

    Restrictions:
    - Limited builtins (no file/network/system access)
    - Execution timeout
    - Restricted imports
    """

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        max_output_lines: int = 1000,
    ):
        self._timeout = timeout_seconds
        self._max_output = max_output_lines

    async def execute(
        self,
        code: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.

        Args:
            code: Python code to execute
            inputs: Input variables to inject

        Returns:
            Execution result with output, errors, and return value
        """
        result: Dict[str, Any] = {
            "success": False,
            "output": [],
            "errors": [],
            "return_value": None,
        }

        # Build safe namespace
        namespace = self._build_namespace(inputs)

        # Capture output
        output_lines: list[str] = []

        def safe_print(*args: Any, **kwargs: Any) -> None:
            if len(output_lines) < self._max_output:
                output_lines.append(" ".join(str(a) for a in args))

        namespace["__builtins__"]["print"] = safe_print

        try:
            await asyncio.wait_for(
                self._run_code(code, namespace),
                timeout=self._timeout,
            )
            result["success"] = True
            result["output"] = output_lines

            # Extract return value if set
            if "_result" in namespace:
                result["return_value"] = namespace["_result"]

        except asyncio.TimeoutError:
            result["errors"].append(f"Execution timed out after {self._timeout}s")
        except Exception as e:
            result["errors"].append(f"{type(e).__name__}: {e}")

        return result

    async def _run_code(self, code: str, namespace: Dict[str, Any]) -> None:
        """Run code in the given namespace."""
        compiled = compile(code, "<sandbox>", "exec")
        exec(compiled, namespace)

    def _build_namespace(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a restricted namespace."""
        import builtins

        safe_builtins_dict = {
            name: getattr(builtins, name)
            for name in _SAFE_BUILTINS
            if hasattr(builtins, name)
        }

        namespace: Dict[str, Any] = {
            "__builtins__": safe_builtins_dict,
            "__name__": "__sandbox__",
        }

        if inputs:
            namespace.update(inputs)

        return namespace
