"""
AION Contract Builder - Generate API contracts from specifications.

Builds typed contracts (input/output schemas) for synthesized systems,
enabling validation and documentation generation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from aion.nlp.types import (
    APISpecification,
    ParameterSpec,
    ToolSpecification,
    WorkflowSpecification,
)


class ContractBuilder:
    """Builds input/output contracts for specifications."""

    @staticmethod
    def build_tool_contract(spec: ToolSpecification) -> Dict[str, Any]:
        """Build a contract for a tool specification."""
        input_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param in spec.parameters:
            input_schema["properties"][param.name] = {
                "type": _python_type_to_json(param.type),
                "description": param.description,
            }
            if param.default is not None:
                input_schema["properties"][param.name]["default"] = param.default
            if param.constraints:
                input_schema["properties"][param.name].update(
                    _constraints_to_json(param.constraints)
                )
            if param.required:
                input_schema["required"].append(param.name)

        output_schema = {
            "type": _python_type_to_json(spec.return_type),
            "description": spec.return_description,
        }

        return {
            "name": spec.name,
            "description": spec.description,
            "input": input_schema,
            "output": output_schema,
            "idempotent": spec.idempotent,
            "timeout": spec.timeout_seconds,
        }

    @staticmethod
    def build_workflow_contract(spec: WorkflowSpecification) -> Dict[str, Any]:
        """Build a contract for a workflow specification."""
        input_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param in spec.inputs:
            input_schema["properties"][param.name] = {
                "type": _python_type_to_json(param.type),
                "description": param.description,
            }
            if param.required:
                input_schema["required"].append(param.name)

        return {
            "name": spec.name,
            "description": spec.description,
            "trigger": {
                "type": spec.trigger_type,
                "config": spec.trigger_config,
            },
            "input": input_schema,
            "steps": [s.to_dict() for s in spec.steps],
            "error_handling": {
                "on_error": spec.on_error,
                "max_retries": spec.max_retries,
            },
        }

    @staticmethod
    def build_api_contract(spec: APISpecification) -> Dict[str, Any]:
        """Build an OpenAPI-like contract for an API specification."""
        paths: Dict[str, Any] = {}

        for endpoint in spec.endpoints:
            path_key = f"{spec.base_path}/{spec.version}{endpoint.path}"
            if path_key not in paths:
                paths[path_key] = {}

            method = endpoint.method.lower()
            paths[path_key][method] = {
                "summary": endpoint.description,
                "parameters": [p.to_dict() for p in endpoint.parameters],
                "responses": {
                    str(code): {"description": desc}
                    for code, desc in endpoint.status_codes.items()
                },
            }

            if endpoint.request_body:
                paths[path_key][method]["requestBody"] = {
                    "content": {
                        "application/json": {"schema": endpoint.request_body}
                    }
                }

            if endpoint.response_schema:
                paths[path_key][method]["responses"]["200"]["content"] = {
                    "application/json": {"schema": endpoint.response_schema}
                }

        return {
            "openapi": "3.0.0",
            "info": {
                "title": spec.name,
                "description": spec.description,
                "version": spec.version,
            },
            "paths": paths,
            "security": [{"bearerAuth": []}] if spec.auth_type else [],
        }


def _python_type_to_json(type_str: str) -> str:
    """Convert Python type string to JSON Schema type."""
    mapping = {
        "string": "string", "str": "string",
        "int": "integer", "integer": "integer",
        "float": "number", "number": "number",
        "bool": "boolean", "boolean": "boolean",
        "list": "array", "array": "array",
        "dict": "object", "object": "object",
        "any": "object", "Any": "object",
        "datetime": "string", "date": "string",
        "url": "string", "email": "string", "path": "string",
    }
    return mapping.get(type_str, "object")


def _constraints_to_json(constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Convert constraint dict to JSON Schema constraints."""
    result: Dict[str, Any] = {}
    mappings = {
        "min": "minimum", "max": "maximum",
        "min_length": "minLength", "max_length": "maxLength",
        "pattern": "pattern",
        "enum": "enum",
    }
    for key, value in constraints.items():
        if key in mappings:
            result[mappings[key]] = value
    return result
