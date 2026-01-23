"""
AION MCP Schema Validation

JSON Schema validation for MCP tool arguments.
Provides:
- Schema-based argument validation
- Type coercion
- Default value handling
- Detailed error messages
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Union

import structlog

logger = structlog.get_logger(__name__)


# Try to import jsonschema for full validation
try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None
    Draft7Validator = None
    JsonSchemaValidationError = Exception


# ============================================
# Validation Errors
# ============================================

@dataclass
class ValidationError:
    """A single validation error."""
    path: str
    message: str
    value: Any = None
    expected: Optional[str] = None

    def __str__(self) -> str:
        if self.path:
            return f"{self.path}: {self.message}"
        return self.message


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    coerced_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [
                {"path": e.path, "message": e.message, "expected": e.expected}
                for e in self.errors
            ],
        }


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, errors: List[ValidationError]):
        self.errors = errors
        message = "; ".join(str(e) for e in errors)
        super().__init__(f"Schema validation failed: {message}")


# ============================================
# Type Coercion
# ============================================

class TypeCoercer:
    """
    Type coercion for MCP arguments.

    Attempts to convert values to expected types where safe.
    """

    @staticmethod
    def coerce_string(value: Any) -> str:
        """Coerce value to string."""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def coerce_integer(value: Any) -> Optional[int]:
        """Coerce value to integer."""
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            try:
                # Handle string representations
                value = value.strip()
                if value.lower() in ("true", "false"):
                    return None
                return int(float(value)) if "." in value else int(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def coerce_number(value: Any) -> Optional[float]:
        """Coerce value to number."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def coerce_boolean(value: Any) -> Optional[bool]:
        """Coerce value to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("true", "yes", "1", "on"):
                return True
            if lower in ("false", "no", "0", "off"):
                return False
            return None
        if isinstance(value, int):
            return bool(value)
        return None

    @staticmethod
    def coerce_array(value: Any) -> Optional[List[Any]]:
        """Coerce value to array."""
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            # Try JSON parsing
            try:
                import json
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # Try comma-separated
            if "," in value:
                return [v.strip() for v in value.split(",")]
        return None

    @staticmethod
    def coerce_object(value: Any) -> Optional[Dict[str, Any]]:
        """Coerce value to object."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                import json
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    @classmethod
    def coerce(cls, value: Any, target_type: str) -> tuple[Any, bool]:
        """
        Coerce value to target type.

        Args:
            value: Value to coerce
            target_type: Target JSON Schema type

        Returns:
            Tuple of (coerced_value, success)
        """
        if value is None:
            return None, True

        coercers = {
            "string": cls.coerce_string,
            "integer": cls.coerce_integer,
            "number": cls.coerce_number,
            "boolean": cls.coerce_boolean,
            "array": cls.coerce_array,
            "object": cls.coerce_object,
        }

        coercer = coercers.get(target_type)
        if coercer is None:
            return value, True  # Unknown type, pass through

        result = coercer(value)
        if result is None and value is not None:
            return value, False

        return result, True


# ============================================
# Schema Validator
# ============================================

class SchemaValidator:
    """
    JSON Schema validator for MCP tool arguments.

    Features:
    - Full JSON Schema Draft 7 support (with jsonschema)
    - Fallback basic validation (without jsonschema)
    - Type coercion
    - Default value handling
    - Detailed error messages
    """

    def __init__(
        self,
        coerce_types: bool = True,
        apply_defaults: bool = True,
        strict: bool = False,
    ):
        """
        Initialize validator.

        Args:
            coerce_types: Whether to attempt type coercion
            apply_defaults: Whether to apply default values
            strict: Whether to reject unknown properties
        """
        self.coerce_types = coerce_types
        self.apply_defaults = apply_defaults
        self.strict = strict

    def validate(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema: JSON Schema

        Returns:
            ValidationResult with status and errors
        """
        # Make a copy for potential modifications
        data = dict(data) if data else {}

        # Apply defaults first
        if self.apply_defaults:
            data = self._apply_defaults(data, schema)

        # Use jsonschema if available
        if JSONSCHEMA_AVAILABLE:
            return self._validate_with_jsonschema(data, schema)

        # Fallback to basic validation
        return self._validate_basic(data, schema)

    def validate_or_raise(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate data and raise on error.

        Args:
            data: Data to validate
            schema: JSON Schema

        Returns:
            Validated (and possibly coerced) data

        Raises:
            SchemaValidationError: If validation fails
        """
        result = self.validate(data, schema)

        if not result.valid:
            raise SchemaValidationError(result.errors)

        return result.coerced_data or data

    def _apply_defaults(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply default values from schema."""
        if schema.get("type") != "object":
            return data

        properties = schema.get("properties", {})
        result = dict(data)

        for prop_name, prop_schema in properties.items():
            if prop_name not in result and "default" in prop_schema:
                result[prop_name] = prop_schema["default"]

        return result

    def _validate_with_jsonschema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> ValidationResult:
        """Validate using jsonschema library."""
        errors = []

        # Coerce types if enabled
        if self.coerce_types:
            data, coercion_errors = self._coerce_data(data, schema)
            errors.extend(coercion_errors)

        # Validate with jsonschema
        validator = Draft7Validator(schema)
        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.path) or "root"
            errors.append(ValidationError(
                path=path,
                message=error.message,
                value=error.instance,
            ))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            coerced_data=data,
        )

    def _validate_basic(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> ValidationResult:
        """Basic validation without jsonschema."""
        errors = []

        # Coerce types if enabled
        if self.coerce_types:
            data, coercion_errors = self._coerce_data(data, schema)
            errors.extend(coercion_errors)

        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data or data[prop] is None:
                errors.append(ValidationError(
                    path=prop,
                    message=f"Required property '{prop}' is missing",
                    expected="present",
                ))

        # Validate property types
        properties = schema.get("properties", {})
        for prop_name, value in data.items():
            if prop_name in properties:
                prop_errors = self._validate_property(
                    prop_name, value, properties[prop_name]
                )
                errors.extend(prop_errors)
            elif self.strict:
                errors.append(ValidationError(
                    path=prop_name,
                    message=f"Unknown property '{prop_name}'",
                ))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            coerced_data=data,
        )

    def _validate_property(
        self,
        name: str,
        value: Any,
        prop_schema: Dict[str, Any],
    ) -> List[ValidationError]:
        """Validate a single property."""
        errors = []

        if value is None:
            return errors

        prop_type = prop_schema.get("type")

        # Type validation
        if prop_type:
            valid_type = self._check_type(value, prop_type)
            if not valid_type:
                errors.append(ValidationError(
                    path=name,
                    message=f"Expected type '{prop_type}', got '{type(value).__name__}'",
                    value=value,
                    expected=prop_type,
                ))

        # Enum validation
        enum_values = prop_schema.get("enum")
        if enum_values and value not in enum_values:
            errors.append(ValidationError(
                path=name,
                message=f"Value must be one of: {enum_values}",
                value=value,
                expected=f"one of {enum_values}",
            ))

        # String constraints
        if prop_type == "string" and isinstance(value, str):
            min_length = prop_schema.get("minLength")
            max_length = prop_schema.get("maxLength")
            pattern = prop_schema.get("pattern")

            if min_length is not None and len(value) < min_length:
                errors.append(ValidationError(
                    path=name,
                    message=f"String length must be at least {min_length}",
                    value=value,
                ))

            if max_length is not None and len(value) > max_length:
                errors.append(ValidationError(
                    path=name,
                    message=f"String length must be at most {max_length}",
                    value=value,
                ))

            if pattern and not re.match(pattern, value):
                errors.append(ValidationError(
                    path=name,
                    message=f"String must match pattern: {pattern}",
                    value=value,
                ))

        # Number constraints
        if prop_type in ("integer", "number") and isinstance(value, (int, float)):
            minimum = prop_schema.get("minimum")
            maximum = prop_schema.get("maximum")
            exclusive_min = prop_schema.get("exclusiveMinimum")
            exclusive_max = prop_schema.get("exclusiveMaximum")

            if minimum is not None and value < minimum:
                errors.append(ValidationError(
                    path=name,
                    message=f"Value must be at least {minimum}",
                    value=value,
                ))

            if maximum is not None and value > maximum:
                errors.append(ValidationError(
                    path=name,
                    message=f"Value must be at most {maximum}",
                    value=value,
                ))

            if exclusive_min is not None and value <= exclusive_min:
                errors.append(ValidationError(
                    path=name,
                    message=f"Value must be greater than {exclusive_min}",
                    value=value,
                ))

            if exclusive_max is not None and value >= exclusive_max:
                errors.append(ValidationError(
                    path=name,
                    message=f"Value must be less than {exclusive_max}",
                    value=value,
                ))

        # Array constraints
        if prop_type == "array" and isinstance(value, list):
            min_items = prop_schema.get("minItems")
            max_items = prop_schema.get("maxItems")
            unique_items = prop_schema.get("uniqueItems")

            if min_items is not None and len(value) < min_items:
                errors.append(ValidationError(
                    path=name,
                    message=f"Array must have at least {min_items} items",
                    value=value,
                ))

            if max_items is not None and len(value) > max_items:
                errors.append(ValidationError(
                    path=name,
                    message=f"Array must have at most {max_items} items",
                    value=value,
                ))

            if unique_items:
                seen: Set[Any] = set()
                for item in value:
                    try:
                        item_hash = hash(str(item))
                        if item_hash in seen:
                            errors.append(ValidationError(
                                path=name,
                                message="Array items must be unique",
                                value=value,
                            ))
                            break
                        seen.add(item_hash)
                    except TypeError:
                        pass

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }

        checker = type_checks.get(expected_type)
        if checker:
            return checker(value)

        return True  # Unknown type

    def _coerce_data(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[ValidationError]]:
        """Coerce data types according to schema."""
        errors = []
        result = dict(data)

        properties = schema.get("properties", {})

        for prop_name, value in data.items():
            if prop_name not in properties:
                continue

            prop_schema = properties[prop_name]
            prop_type = prop_schema.get("type")

            if prop_type and value is not None:
                coerced, success = TypeCoercer.coerce(value, prop_type)
                if success:
                    result[prop_name] = coerced
                else:
                    errors.append(ValidationError(
                        path=prop_name,
                        message=f"Cannot coerce '{type(value).__name__}' to '{prop_type}'",
                        value=value,
                        expected=prop_type,
                    ))

        return result, errors


# ============================================
# Tool Argument Validator
# ============================================

class ToolArgumentValidator:
    """
    Specialized validator for MCP tool arguments.

    Provides:
    - Schema extraction from Tool definitions
    - Argument preparation with defaults
    - Validation with helpful errors
    """

    def __init__(
        self,
        coerce_types: bool = True,
        apply_defaults: bool = True,
    ):
        """
        Initialize validator.

        Args:
            coerce_types: Whether to coerce argument types
            apply_defaults: Whether to apply default values
        """
        self._validator = SchemaValidator(
            coerce_types=coerce_types,
            apply_defaults=apply_defaults,
        )

    def validate_tool_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        input_schema: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate arguments for a tool.

        Args:
            tool_name: Tool name (for error messages)
            arguments: Arguments to validate
            input_schema: Tool's input schema

        Returns:
            ValidationResult
        """
        result = self._validator.validate(arguments, input_schema)

        # Add tool name to error paths
        for error in result.errors:
            error.path = f"{tool_name}.{error.path}" if error.path else tool_name

        return result

    def prepare_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        input_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare and validate tool arguments.

        Args:
            tool_name: Tool name
            arguments: Raw arguments
            input_schema: Tool's input schema

        Returns:
            Validated and prepared arguments

        Raises:
            SchemaValidationError: If validation fails
        """
        result = self.validate_tool_arguments(tool_name, arguments, input_schema)

        if not result.valid:
            raise SchemaValidationError(result.errors)

        return result.coerced_data or arguments


# Global validator instance
_validator = ToolArgumentValidator()


def validate_tool_arguments(
    tool_name: str,
    arguments: Dict[str, Any],
    input_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience function to validate tool arguments.

    Args:
        tool_name: Tool name
        arguments: Arguments to validate
        input_schema: Tool's input schema

    Returns:
        Validated arguments

    Raises:
        SchemaValidationError: If validation fails
    """
    return _validator.prepare_arguments(tool_name, arguments, input_schema)
