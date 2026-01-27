"""
AION Base Synthesizer - Abstract foundation for all code synthesizers.

Provides shared functionality for code generation including
LLM interaction, code formatting, import resolution, and test scaffolding.
"""

from __future__ import annotations

import re
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import GeneratedCode, SpecificationType

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel
    from aion.nlp.config import SynthesisConfig

logger = structlog.get_logger(__name__)


# Python type mappings
TYPE_MAP: Dict[str, str] = {
    "string": "str",
    "str": "str",
    "number": "float",
    "float": "float",
    "integer": "int",
    "int": "int",
    "boolean": "bool",
    "bool": "bool",
    "array": "List[Any]",
    "list": "List[Any]",
    "object": "Dict[str, Any]",
    "dict": "Dict[str, Any]",
    "any": "Any",
    "Any": "Any",
    "datetime": "datetime",
    "url": "str",
    "email": "str",
    "path": "str",
    "format": "str",
}


class BaseSynthesizer(ABC):
    """
    Abstract base for all AION code synthesizers.

    Provides:
    - LLM-powered code generation with retry
    - Code cleanup and formatting
    - Import resolution
    - Type mapping
    - Test generation scaffolding
    """

    def __init__(self, kernel: AIONKernel, config: Optional[SynthesisConfig] = None):
        self.kernel = kernel
        self._config = config

    @abstractmethod
    async def synthesize(self, spec: Any) -> GeneratedCode:
        """Generate code from a specification."""
        ...

    # =========================================================================
    # LLM Code Generation
    # =========================================================================

    async def _llm_generate(
        self,
        prompt: str,
        max_retries: Optional[int] = None,
    ) -> str:
        """Generate code using LLM with exponential backoff + jitter retry."""
        import asyncio
        import random

        if max_retries is None:
            max_retries = (
                self._config.max_generation_retries
                if self._config and hasattr(self._config, "max_generation_retries")
                else 3
            )

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = await self.kernel.llm.complete(
                    [{"role": "user", "content": prompt}]
                )
                content = response.content if hasattr(response, "content") else str(response)
                return self._clean_code_response(content)
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM generation attempt failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter to prevent thundering herd
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)

        if last_error:
            raise last_error
        return ""

    # =========================================================================
    # Code Utilities
    # =========================================================================

    def _clean_code_response(self, response: str) -> str:
        """Clean LLM code response by removing markdown fences."""
        code = response.strip()

        # Remove markdown code blocks
        if code.startswith("```"):
            lines = code.split("\n")
            # Find end of first fence
            start = 1
            # Find start of last fence
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            code = "\n".join(lines[start:end])

        return code.strip()

    def _indent(self, code: str, spaces: int = 4) -> str:
        """Indent code by specified spaces."""
        return textwrap.indent(code, " " * spaces)

    def _python_type(self, type_str: str) -> str:
        """Convert type string to Python type annotation."""
        return TYPE_MAP.get(type_str.lower(), "Any")

    def _resolve_imports(self, code: str) -> List[str]:
        """Determine required imports from code content."""
        imports: List[str] = []

        import_checks = [
            ("aiohttp", "import aiohttp"),
            ("json.", "import json"),
            ("json.loads", "import json"),
            ("json.dumps", "import json"),
            ("datetime", "from datetime import datetime, timedelta"),
            ("re.", "import re"),
            ("re.search", "import re"),
            ("asyncio", "import asyncio"),
            ("hashlib", "import hashlib"),
            ("uuid", "import uuid"),
            ("pathlib", "from pathlib import Path"),
            ("logging", "import logging"),
            ("dataclass", "from dataclasses import dataclass, field"),
        ]

        seen = set()
        for pattern, import_stmt in import_checks:
            if pattern in code and import_stmt not in seen:
                imports.append(import_stmt)
                seen.add(import_stmt)

        # Always include typing
        type_patterns = ["Any", "Dict", "List", "Optional", "Union", "Tuple", "Set"]
        used_types = [t for t in type_patterns if t in code]
        if used_types:
            imports.append(f"from typing import {', '.join(sorted(used_types))}")

        return imports

    def _generate_validation_code(self, parameters: List[Dict[str, Any]]) -> str:
        """Generate parameter validation code."""
        lines: List[str] = []

        for p in parameters:
            name = p.get("name", "")
            required = p.get("required", True)
            param_type = p.get("type", "string")

            if required:
                lines.append(
                    f"if {name} is None:\n"
                    f"    raise ValueError('{name} is required')"
                )

            if param_type in ("string", "str") and p.get("min_length"):
                min_len = p["min_length"]
                lines.append(
                    f"if {name} is not None and len({name}) < {min_len}:\n"
                    f"    raise ValueError('{name} must be at least {min_len} characters')"
                )

            if param_type in ("int", "float", "number"):
                if p.get("min") is not None:
                    min_val = p['min']
                    lines.append(
                        f"if {name} is not None and {name} < {min_val}:\n"
                        f"    raise ValueError('{name} must be >= {min_val}')"
                    )
                if p.get("max") is not None:
                    max_val = p['max']
                    lines.append(
                        f"if {name} is not None and {name} > {max_val}:\n"
                        f"    raise ValueError('{name} must be <= {max_val}')"
                    )

        return "\n".join(lines) if lines else "pass  # No validation needed"

    def _build_function_signature(
        self,
        name: str,
        parameters: List[Dict[str, Any]],
        return_type: str = "Any",
        is_async: bool = True,
    ) -> str:
        """Build a function signature string."""
        params: List[str] = []

        for p in parameters:
            param_str = p.get("name", "param")
            if p.get("type"):
                param_str += f": {self._python_type(p['type'])}"
            if p.get("default") is not None:
                param_str += f" = {repr(p['default'])}"
            elif not p.get("required", True):
                param_str += " = None"
            params.append(param_str)

        params_str = ", ".join(params)
        ret_type = self._python_type(return_type)
        prefix = "async def" if is_async else "def"

        return f"{prefix} {name}({params_str}) -> {ret_type}:"

    def _build_docstring(
        self,
        description: str,
        parameters: List[Dict[str, Any]],
        return_type: str = "Any",
        return_description: str = "",
    ) -> str:
        """Build a Python docstring."""
        parts = [f'"""{description}']

        if parameters:
            parts.append("")
            parts.append("Args:")
            for p in parameters:
                pname = p.get("name", "param")
                pdesc = p.get("description", "")
                ptype = p.get("type", "Any")
                parts.append(f"    {pname} ({ptype}): {pdesc}")

        parts.append("")
        parts.append("Returns:")
        parts.append(f"    {return_description or return_type}")

        parts.append('"""')
        return "\n".join(parts)
