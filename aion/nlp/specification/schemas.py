"""
AION Specification Schemas - JSON Schema definitions for specs.

Provides validation schemas for each specification type to ensure
well-formed specs before synthesis.
"""

from __future__ import annotations

from typing import Any, Dict


class SpecificationSchemas:
    """JSON Schema definitions for AION specifications."""

    TOOL_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "required": ["name", "description"],
        "properties": {
            "name": {"type": "string", "minLength": 1, "pattern": "^[a-z_][a-z0-9_]*$"},
            "description": {"type": "string", "minLength": 1},
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "type"],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": [
                            "string", "int", "float", "bool", "list",
                            "dict", "Any", "datetime", "url", "email", "path",
                        ]},
                        "description": {"type": "string"},
                        "required": {"type": "boolean"},
                        "default": {},
                    },
                },
            },
            "return_type": {"type": "string"},
            "api_endpoint": {"type": ["string", "null"], "format": "uri"},
            "api_method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
        },
    }

    WORKFLOW_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "required": ["name", "description"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "trigger_type": {
                "type": "string",
                "enum": ["manual", "schedule", "event", "webhook"],
            },
            "trigger_config": {"type": "object"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "name", "action"],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "action": {"type": "string"},
                        "params": {"type": "object"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "on_error": {"type": "string", "enum": ["stop", "continue", "retry"]},
            "max_retries": {"type": "integer", "minimum": 0},
        },
    }

    AGENT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "required": ["name", "description"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "system_prompt": {"type": "string"},
            "personality_traits": {"type": "array", "items": {"type": "string"}},
            "primary_goal": {"type": "string"},
            "allowed_tools": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "max_iterations": {"type": "integer", "minimum": 1},
        },
    }

    API_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "required": ["name", "description"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "base_path": {"type": "string"},
            "endpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["path", "method"],
                    "properties": {
                        "path": {"type": "string"},
                        "method": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
        },
    }

    @classmethod
    def get_schema(cls, spec_type: str) -> Dict[str, Any]:
        """Get the schema for a specification type."""
        schemas = {
            "tool": cls.TOOL_SCHEMA,
            "workflow": cls.WORKFLOW_SCHEMA,
            "agent": cls.AGENT_SCHEMA,
            "api": cls.API_SCHEMA,
        }
        return schemas.get(spec_type, {})
