"""
AION Code Generator - General-purpose code generation utilities.

Provides LLM-based code generation for arbitrary code tasks,
including function generation, refactoring, and code completion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.synthesis.base import BaseSynthesizer
from aion.nlp.types import GeneratedCode, SpecificationType

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class CodeGenerator(BaseSynthesizer):
    """
    General-purpose code generator for NLP programming.

    Used for:
    - Generating arbitrary functions
    - Modifying existing code
    - Creating utility code
    - Code completion
    """

    async def synthesize(self, spec: Any) -> GeneratedCode:
        """Generate code from a generic specification dict."""
        if isinstance(spec, dict):
            return await self._generate_from_dict(spec)
        raise ValueError(f"CodeGenerator expects dict spec, got {type(spec)}")

    async def _generate_from_dict(self, spec: Dict[str, Any]) -> GeneratedCode:
        """Generate code from a spec dictionary."""
        name = spec.get("name", "generated_function")
        description = spec.get("description", "")
        language = spec.get("language", "python")

        prompt = f"""Generate a complete {language} implementation:

Name: {name}
Description: {description}
Parameters: {spec.get("parameters", [])}
Return type: {spec.get("return_type", "Any")}
Additional context: {spec.get("context", "")}

Requirements:
- Complete, working implementation
- Type hints and docstring
- Error handling
- Be concise and practical

Generate ONLY the code:"""

        code = await self._llm_generate(prompt)
        imports = self._resolve_imports(code)

        return GeneratedCode(
            language=language,
            code=code,
            filename=f"{name}.py",
            spec_type=SpecificationType.FUNCTION,
            imports=imports,
            docstring=description,
        )

    async def generate_function(
        self,
        name: str,
        description: str,
        parameters: List[Dict[str, Any]],
        return_type: str = "Any",
        is_async: bool = True,
    ) -> str:
        """Generate a single function."""
        prompt = f"""Generate a Python {'async ' if is_async else ''}function:

Function: {name}
Description: {description}
Parameters: {parameters}
Return type: {return_type}

Generate only the function with docstring and implementation:"""

        return await self._llm_generate(prompt)

    async def modify_code(
        self,
        original_code: str,
        modification: str,
    ) -> str:
        """Modify existing code based on instructions."""
        prompt = f"""Modify this code according to the instructions:

Original code:
```python
{original_code}
```

Modification requested: {modification}

Return ONLY the modified code:"""

        return await self._llm_generate(prompt)

    async def generate_tests(
        self,
        code: str,
        description: str,
    ) -> str:
        """Generate tests for given code."""
        prompt = f"""Generate pytest test cases for this code:

```python
{code}
```

Description: {description}

Requirements:
- Use pytest and pytest.mark.asyncio for async tests
- Test happy path and error cases
- Be practical and concise

Generate ONLY the test code:"""

        return await self._llm_generate(prompt)
