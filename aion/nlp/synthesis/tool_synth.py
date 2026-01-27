"""
AION Tool Synthesizer - Generate tool implementations from specifications.

Creates production-ready async Python tools with:
- Full type annotations
- Parameter validation
- Error handling with retries
- API integration support
- Comprehensive test generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.synthesis.base import BaseSynthesizer
from aion.nlp.types import (
    GeneratedCode,
    SpecificationType,
    ToolSpecification,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class ToolSynthesizer(BaseSynthesizer):
    """
    Synthesizes tool code from ToolSpecification.

    Generates:
    - Async function implementation
    - Parameter validation
    - API client code (if needed)
    - Error handling with retries
    - Test suite
    """

    async def synthesize(self, spec: ToolSpecification) -> GeneratedCode:
        """Generate tool code from specification."""
        # Generate main implementation
        code = await self._generate_implementation(spec)

        # Generate tests
        test_code = self._generate_tests(spec)

        # Resolve imports
        imports = self._resolve_imports(code)

        return GeneratedCode(
            language="python",
            code=code,
            filename=f"{spec.name}.py",
            spec_type=SpecificationType.TOOL,
            imports=imports,
            dependencies=spec.dependencies,
            test_code=test_code,
            docstring=spec.description,
        )

    async def _generate_implementation(self, spec: ToolSpecification) -> str:
        """Generate the complete tool implementation."""
        # Build parameter list for signature
        params_dicts = [
            {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required,
                "default": p.default,
            }
            for p in spec.parameters
        ]

        # Function signature
        signature = self._build_function_signature(
            name=spec.name,
            parameters=params_dicts,
            return_type=spec.return_type,
            is_async=True,
        )

        # Docstring
        docstring = self._build_docstring(
            description=spec.description,
            parameters=params_dicts,
            return_type=spec.return_type,
            return_description=spec.return_description,
        )

        # Validation
        validation = self._generate_validation_code(params_dicts)

        # Implementation body
        if spec.api_endpoint:
            body = self._generate_api_body(spec)
        else:
            body = await self._generate_logic_body(spec)

        # Combine
        code = f"""{signature}
    {docstring}

    # Parameter validation
    {self._indent(validation, 0)}

    # Implementation
{self._indent(body, 4)}
"""
        return code.strip()

    def _generate_api_body(self, spec: ToolSpecification) -> str:
        """Generate implementation for API-backed tools."""
        headers_init = "headers: Dict[str, str] = {}"
        if spec.api_headers:
            headers_init = f"headers: Dict[str, str] = {repr(spec.api_headers)}"

        auth_block = ""
        if spec.auth_required:
            auth_block = """
# Authentication
import os
auth_token = os.environ.get("API_TOKEN", "")
if auth_token:
    headers["Authorization"] = f"Bearer {auth_token}"
"""

        # Build request parameters
        param_names = [p.name for p in spec.parameters]
        if spec.api_method in ("GET", "DELETE"):
            params_block = f"params={{k: v for k, v in {{{', '.join(repr(n) + ': ' + n for n in param_names)}}}.items() if v is not None}}"
            request_kwargs = f"""
    url="{spec.api_endpoint}",
    headers=headers,
    params={params_block},"""
        else:
            body_block = f"json_body={{k: v for k, v in {{{', '.join(repr(n) + ': ' + n for n in param_names)}}}.items() if v is not None}}"
            request_kwargs = f"""
    url="{spec.api_endpoint}",
    headers=headers,
    json={body_block},"""

        retry_block = ""
        if spec.retry_on_failure:
            retry_block = f"""
max_retries = {spec.max_retries}
for attempt in range(max_retries + 1):
    try:
"""
            indent = "        "
        else:
            indent = ""

        request_code = f"""{indent}async with aiohttp.ClientSession() as session:
{indent}    async with session.{spec.api_method.lower()}({request_kwargs.strip()}
{indent}    ) as response:
{indent}        response.raise_for_status()
{indent}        return await response.json()"""

        if spec.retry_on_failure:
            error_handling = f"""    except aiohttp.ClientError as e:
        if attempt == max_retries:
            raise RuntimeError(f"API request failed after {{max_retries + 1}} attempts: {{e}}")
        await asyncio.sleep(2 ** attempt)  # Exponential backoff"""
        else:
            error_handling = ""

        return f"""{headers_init}
{auth_block}
try:
{retry_block}{request_code}
{error_handling}
except Exception as e:
    raise RuntimeError(f"Tool '{spec.name}' failed: {{e}}")"""

    async def _generate_logic_body(self, spec: ToolSpecification) -> str:
        """Use LLM to generate implementation logic."""
        prompt = f"""Generate Python implementation for this tool function body.

Name: {spec.name}
Description: {spec.description}
Parameters: {[(p.name, p.type, p.description) for p in spec.parameters]}
Return type: {spec.return_type}
Implementation notes: {spec.implementation_notes}

Requirements:
- Generate ONLY the function body (no def statement or docstring)
- Use async/await for any I/O operations
- Include proper error handling
- Return the correct type
- Be concise and practical
- Do NOT include imports

Generate clean Python code:"""

        try:
            code = await self._llm_generate(prompt)
            # Ensure proper indentation
            lines = code.strip().split("\n")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("LLM tool generation failed, using stub", error=str(e))
            return f"# TODO: Implement {spec.name}\nraise NotImplementedError('{spec.name} not yet implemented')"

    def _generate_tests(self, spec: ToolSpecification) -> str:
        """Generate test suite for the tool."""
        test_cases: List[str] = []

        # Basic invocation test
        default_args = []
        for p in spec.parameters:
            if p.required:
                if p.type in ("string", "str"):
                    default_args.append(f'{p.name}="test"')
                elif p.type in ("int", "integer"):
                    default_args.append(f"{p.name}=1")
                elif p.type in ("float", "number"):
                    default_args.append(f"{p.name}=1.0")
                elif p.type in ("bool", "boolean"):
                    default_args.append(f"{p.name}=True")
                elif p.type in ("list", "array"):
                    default_args.append(f"{p.name}=[]")
                elif p.type in ("dict", "object"):
                    default_args.append(f"{p.name}={{}}")
                else:
                    default_args.append(f'{p.name}="test"')

        args_str = ", ".join(default_args)

        test_cases.append(f"""
@pytest.mark.asyncio
async def test_{spec.name}_basic():
    \"\"\"Test basic invocation of {spec.name}.\"\"\"
    result = await {spec.name}({args_str})
    assert result is not None
""")

        # Required parameter tests
        for p in spec.parameters:
            if p.required:
                test_cases.append(f"""
@pytest.mark.asyncio
async def test_{spec.name}_missing_{p.name}():
    \"\"\"Test that missing {p.name} raises ValueError.\"\"\"
    with pytest.raises((ValueError, TypeError)):
        await {spec.name}({p.name}=None)
""")

        return f"""import pytest
from {spec.name} import {spec.name}

{"".join(test_cases)}"""
