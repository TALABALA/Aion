"""
AION Specification Validator - Validate specifications before synthesis.

Ensures specifications are well-formed and contain all required
information before code generation begins.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from aion.nlp.types import (
    APISpecification,
    AgentSpecification,
    IntegrationSpecification,
    Specification,
    ToolSpecification,
    WorkflowSpecification,
)

logger = structlog.get_logger(__name__)


class SpecValidationResult:
    """Result of specification validation."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class SpecValidator:
    """Validates specifications for completeness and correctness."""

    def validate(self, spec: Specification) -> SpecValidationResult:
        """Validate a specification."""
        if isinstance(spec, ToolSpecification):
            return self._validate_tool(spec)
        elif isinstance(spec, WorkflowSpecification):
            return self._validate_workflow(spec)
        elif isinstance(spec, AgentSpecification):
            return self._validate_agent(spec)
        elif isinstance(spec, APISpecification):
            return self._validate_api(spec)
        elif isinstance(spec, IntegrationSpecification):
            return self._validate_integration(spec)
        else:
            result = SpecValidationResult()
            result.add_error(f"Unknown specification type: {type(spec).__name__}")
            return result

    def _validate_tool(self, spec: ToolSpecification) -> SpecValidationResult:
        result = SpecValidationResult()

        if not spec.name:
            result.add_error("Tool must have a name")
        if not spec.description:
            result.add_error("Tool must have a description")

        # Check parameter names are unique
        names = [p.name for p in spec.parameters]
        if len(names) != len(set(names)):
            result.add_error("Tool parameters must have unique names")

        # Check for valid Python identifier
        if spec.name and not spec.name.isidentifier():
            result.add_error(f"Tool name '{spec.name}' is not a valid Python identifier")

        # Warnings
        if not spec.parameters:
            result.add_warning("Tool has no parameters defined")
        if spec.api_endpoint and not spec.api_endpoint.startswith(("http://", "https://")):
            result.add_warning(f"API endpoint may be invalid: {spec.api_endpoint}")

        return result

    def _validate_workflow(self, spec: WorkflowSpecification) -> SpecValidationResult:
        result = SpecValidationResult()

        if not spec.name:
            result.add_error("Workflow must have a name")
        if not spec.description:
            result.add_error("Workflow must have a description")

        # Validate steps
        step_ids = set()
        for step in spec.steps:
            if step.id in step_ids:
                result.add_error(f"Duplicate step ID: {step.id}")
            step_ids.add(step.id)

            # Check dependencies reference valid steps
            for dep in step.depends_on:
                if dep not in step_ids:
                    result.add_warning(f"Step '{step.id}' depends on unknown step '{dep}'")

        if not spec.steps:
            result.add_warning("Workflow has no steps defined")
        if spec.trigger_type == "schedule" and not spec.trigger_config:
            result.add_warning("Schedule trigger has no configuration")

        return result

    def _validate_agent(self, spec: AgentSpecification) -> SpecValidationResult:
        result = SpecValidationResult()

        if not spec.name:
            result.add_error("Agent must have a name")
        if not spec.description:
            result.add_error("Agent must have a description")

        if not spec.primary_goal:
            result.add_warning("Agent has no primary goal defined")
        if not spec.system_prompt:
            result.add_warning("Agent has no system prompt")
        if spec.max_iterations < 1:
            result.add_error("Agent max_iterations must be >= 1")

        return result

    def _validate_api(self, spec: APISpecification) -> SpecValidationResult:
        result = SpecValidationResult()

        if not spec.name:
            result.add_error("API must have a name")
        if not spec.endpoints:
            result.add_warning("API has no endpoints defined")

        # Check for duplicate endpoint paths + methods
        seen = set()
        for ep in spec.endpoints:
            key = (ep.path, ep.method)
            if key in seen:
                result.add_error(f"Duplicate endpoint: {ep.method} {ep.path}")
            seen.add(key)

        return result

    def _validate_integration(self, spec: IntegrationSpecification) -> SpecValidationResult:
        result = SpecValidationResult()

        if not spec.name:
            result.add_error("Integration must have a name")
        if not spec.source_system:
            result.add_error("Integration must have a source system")
        if not spec.target_system:
            result.add_error("Integration must have a target system")
        if spec.source_system == spec.target_system:
            result.add_warning("Source and target systems are the same")

        return result
