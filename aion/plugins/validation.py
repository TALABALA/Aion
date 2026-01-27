"""
AION Plugin Validation

Validates plugin manifests and code.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import structlog

from aion.plugins.types import (
    PluginManifest,
    PluginPermissions,
    PermissionLevel,
    SemanticVersion,
    ValidationResult,
)

logger = structlog.get_logger(__name__)


# Valid plugin ID pattern
PLUGIN_ID_PATTERN = re.compile(r'^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$')

# Reserved plugin IDs
RESERVED_IDS = {
    "aion", "core", "system", "builtin", "internal",
    "plugin", "plugins", "kernel", "manager",
}

# Maximum sizes
MAX_ID_LENGTH = 64
MAX_NAME_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 2048


class PluginValidator:
    """
    Validates plugin manifests and configurations.

    Features:
    - Manifest validation
    - Permission validation
    - Security checks
    - Configuration schema validation
    """

    def __init__(self, aion_version: Optional[SemanticVersion] = None):
        self._aion_version = aion_version or SemanticVersion(1, 0, 0)
        self._custom_validators: List[callable] = []

    def validate(self, manifest: PluginManifest) -> ValidationResult:
        """
        Validate a plugin manifest.

        Args:
            manifest: Manifest to validate

        Returns:
            ValidationResult with errors, warnings, and info
        """
        result = ValidationResult(valid=True)

        # Run all validations
        self._validate_identity(manifest, result)
        self._validate_version(manifest, result)
        self._validate_entry_point(manifest, result)
        self._validate_permissions(manifest, result)
        self._validate_dependencies(manifest, result)
        self._validate_config_schema(manifest, result)

        # Run custom validators
        for validator in self._custom_validators:
            try:
                validator(manifest, result)
            except Exception as e:
                result.add_warning(f"Custom validator error: {e}")

        return result

    def _validate_identity(
        self,
        manifest: PluginManifest,
        result: ValidationResult,
    ) -> None:
        """Validate plugin identity fields."""
        # ID validation
        if not manifest.id:
            result.add_error("Plugin ID is required")
        elif len(manifest.id) > MAX_ID_LENGTH:
            result.add_error(f"Plugin ID exceeds maximum length of {MAX_ID_LENGTH}")
        elif not PLUGIN_ID_PATTERN.match(manifest.id):
            result.add_error(
                f"Invalid plugin ID format: '{manifest.id}'. "
                "Must be lowercase alphanumeric with dashes/underscores"
            )
        elif manifest.id.lower() in RESERVED_IDS:
            result.add_error(f"Plugin ID '{manifest.id}' is reserved")

        # Name validation
        if not manifest.name:
            result.add_error("Plugin name is required")
        elif len(manifest.name) > MAX_NAME_LENGTH:
            result.add_error(f"Plugin name exceeds maximum length of {MAX_NAME_LENGTH}")

        # Description validation
        if len(manifest.description) > MAX_DESCRIPTION_LENGTH:
            result.add_error(
                f"Plugin description exceeds maximum length of {MAX_DESCRIPTION_LENGTH}"
            )

        # Recommendations
        if not manifest.description:
            result.add_warning("Plugin should have a description")
        if not manifest.author.name:
            result.add_warning("Plugin should specify an author")

    def _validate_version(
        self,
        manifest: PluginManifest,
        result: ValidationResult,
    ) -> None:
        """Validate version fields."""
        # Plugin version
        if manifest.version.major == 0 and manifest.version.minor == 0 and manifest.version.patch == 0:
            result.add_warning("Plugin version is 0.0.0, consider setting a proper version")

        # AION version compatibility
        if not manifest.aion_version.satisfies(self._aion_version):
            result.add_error(
                f"Plugin requires AION {manifest.aion_version}, "
                f"but running {self._aion_version}"
            )

        # Python version (basic check)
        if manifest.python_version and not manifest.python_version.startswith(">="):
            result.add_info("Consider using '>=' for Python version compatibility")

    def _validate_entry_point(
        self,
        manifest: PluginManifest,
        result: ValidationResult,
    ) -> None:
        """Validate entry point."""
        if not manifest.entry_point:
            result.add_warning("No entry point specified, plugin must be discoverable")
            return

        if ":" in manifest.entry_point:
            module, class_name = manifest.entry_point.split(":", 1)
            if not module:
                result.add_error("Entry point module name is empty")
            if not class_name:
                result.add_error("Entry point class name is empty")
            if not class_name[0].isupper():
                result.add_warning(
                    f"Entry point class '{class_name}' should be CamelCase"
                )

    def _validate_permissions(
        self,
        manifest: PluginManifest,
        result: ValidationResult,
    ) -> None:
        """Validate permission requests."""
        permissions = manifest.permissions

        # Check for excessive permissions
        if permissions.level == PermissionLevel.FULL:
            result.add_warning(
                "Plugin requests FULL permissions. This should only be "
                "granted to trusted plugins."
            )

        # Network permissions
        if permissions.network_access:
            if not permissions.allowed_domains and permissions.level != PermissionLevel.FULL:
                result.add_warning(
                    "Plugin requests network access without specifying allowed domains"
                )

        # File system permissions
        if permissions.file_system_access:
            if not permissions.allowed_paths and permissions.level != PermissionLevel.FULL:
                result.add_warning(
                    "Plugin requests file system access without specifying allowed paths"
                )

        # Subprocess permissions
        if permissions.subprocess_access:
            result.add_warning(
                "Plugin requests subprocess execution permission. "
                "Ensure this is necessary."
            )
            if not permissions.allowed_commands:
                result.add_warning(
                    "Plugin requests subprocess access without specifying allowed commands"
                )

        # Resource limits
        limits = permissions.resource_limits
        if limits.max_memory_mb > 1024:
            result.add_warning(
                f"Plugin requests high memory limit: {limits.max_memory_mb}MB"
            )
        if limits.max_execution_time_seconds > 300:
            result.add_warning(
                f"Plugin requests long execution timeout: {limits.max_execution_time_seconds}s"
            )

    def _validate_dependencies(
        self,
        manifest: PluginManifest,
        result: ValidationResult,
    ) -> None:
        """Validate dependencies."""
        seen_deps = set()

        for dep in manifest.dependencies:
            # Check for duplicates
            if dep.plugin_id in seen_deps:
                result.add_error(f"Duplicate dependency: {dep.plugin_id}")
            seen_deps.add(dep.plugin_id)

            # Self-dependency
            if dep.plugin_id == manifest.id:
                result.add_error("Plugin cannot depend on itself")

            # Version constraint validation
            if dep.version_constraint.exact_version and dep.version_constraint.min_version:
                result.add_warning(
                    f"Dependency {dep.plugin_id} has both exact and minimum version"
                )

        # pip dependencies
        for pip_dep in manifest.pip_dependencies:
            if not pip_dep:
                result.add_warning("Empty pip dependency entry")

    def _validate_config_schema(
        self,
        manifest: PluginManifest,
        result: ValidationResult,
    ) -> None:
        """Validate configuration schema."""
        schema = manifest.config_schema

        if not schema:
            return

        # Basic JSON Schema validation
        if "type" not in schema:
            result.add_warning("Config schema should specify a type")

        if schema.get("type") == "object":
            if "properties" not in schema:
                result.add_warning("Object schema should specify properties")

    # === Custom Validators ===

    def add_validator(self, validator: callable) -> None:
        """Add a custom validator function."""
        self._custom_validators.append(validator)

    def remove_validator(self, validator: callable) -> None:
        """Remove a custom validator function."""
        if validator in self._custom_validators:
            self._custom_validators.remove(validator)

    # === Configuration Validation ===

    def validate_config(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate configuration against schema.

        Args:
            config: Configuration to validate
            schema: JSON Schema

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        if not schema:
            return result

        try:
            import jsonschema
            jsonschema.validate(config, schema)
        except ImportError:
            result.add_warning("jsonschema not installed, skipping validation")
        except jsonschema.ValidationError as e:
            result.add_error(f"Configuration validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            result.add_error(f"Invalid schema: {e.message}")

        return result

    # === Security Checks ===

    def security_audit(self, manifest: PluginManifest) -> ValidationResult:
        """
        Perform security audit on plugin manifest.

        Args:
            manifest: Manifest to audit

        Returns:
            ValidationResult with security findings
        """
        result = ValidationResult(valid=True)

        permissions = manifest.permissions

        # High-risk permissions
        high_risk = []
        if permissions.subprocess_access:
            high_risk.append("subprocess_access")
        if permissions.level == PermissionLevel.FULL:
            high_risk.append("full_permissions")
        if permissions.file_system_access and not permissions.allowed_paths:
            high_risk.append("unrestricted_file_access")
        if permissions.network_access and not permissions.allowed_domains:
            high_risk.append("unrestricted_network_access")

        if high_risk:
            result.add_warning(f"High-risk permissions: {', '.join(high_risk)}")

        # Check for potentially dangerous patterns
        if permissions.allowed_paths:
            dangerous_paths = ["/", "/etc", "/usr", "/var", "/home"]
            for path in permissions.allowed_paths:
                if path in dangerous_paths:
                    result.add_warning(f"Potentially dangerous path allowed: {path}")

        if permissions.allowed_domains:
            if "*" in permissions.allowed_domains:
                result.add_warning("Wildcard domain access allowed")

        # Check signature
        if not manifest.signature:
            result.add_info("Plugin is not signed")

        if not manifest.verified:
            result.add_info("Plugin is not verified")

        return result
