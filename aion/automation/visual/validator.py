"""
AION Workflow Validator

Validates workflow definitions and visual graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog

from aion.automation.types import Workflow, WorkflowStep, TriggerType, ActionType

logger = structlog.get_logger(__name__)


@dataclass
class ValidationError:
    """A validation error."""
    code: str
    message: str
    path: Optional[str] = None
    node_id: Optional[str] = None
    severity: str = "error"  # error, warning, info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "node_id": self.node_id,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """Result of workflow validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


class WorkflowValidator:
    """
    Validates workflow definitions.

    Checks:
    - Structural validity
    - Reference integrity
    - Configuration completeness
    - Cycle detection
    - Unreachable step detection
    """

    def __init__(self):
        # Required fields per action type
        self._action_required_fields = {
            ActionType.TOOL: ["tool_name"],
            ActionType.WEBHOOK: ["url"],
            ActionType.AGENT: ["agent_class"],
            ActionType.NOTIFICATION: ["channel", "message"],
            ActionType.WORKFLOW: ["workflow_id"],
        }

        # Required fields per trigger type
        self._trigger_required_fields = {
            TriggerType.SCHEDULE: ["cron_expression"],
            TriggerType.WEBHOOK: ["path"],
            TriggerType.EVENT: ["event_type"],
        }

    def validate(self, workflow: Workflow) -> ValidationResult:
        """
        Validate a workflow definition.

        Args:
            workflow: Workflow to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Basic structure checks
        errors.extend(self._validate_structure(workflow))

        # Trigger validation
        errors.extend(self._validate_triggers(workflow))

        # Step validation
        errors.extend(self._validate_steps(workflow))

        # Reference integrity
        errors.extend(self._validate_references(workflow))

        # Cycle detection
        cycle_errors = self._detect_cycles(workflow)
        errors.extend(cycle_errors)

        # Unreachable steps
        unreachable_warnings = self._find_unreachable_steps(workflow)
        warnings.extend(unreachable_warnings)

        # Configuration completeness
        config_warnings = self._validate_configurations(workflow)
        warnings.extend(config_warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_structure(self, workflow: Workflow) -> List[ValidationError]:
        """Validate basic workflow structure."""
        errors = []

        if not workflow.id:
            errors.append(ValidationError(
                code="MISSING_ID",
                message="Workflow must have an ID",
                path="id",
            ))

        if not workflow.name:
            errors.append(ValidationError(
                code="MISSING_NAME",
                message="Workflow must have a name",
                path="name",
            ))

        if not workflow.steps and not workflow.triggers:
            errors.append(ValidationError(
                code="EMPTY_WORKFLOW",
                message="Workflow must have at least one trigger or step",
            ))

        return errors

    def _validate_triggers(self, workflow: Workflow) -> List[ValidationError]:
        """Validate workflow triggers."""
        errors = []

        for i, trigger in enumerate(workflow.triggers):
            path = f"triggers[{i}]"

            # Check type
            if not trigger.type:
                errors.append(ValidationError(
                    code="MISSING_TRIGGER_TYPE",
                    message="Trigger must have a type",
                    path=path,
                ))
                continue

            # Check required fields
            required = self._trigger_required_fields.get(trigger.type, [])
            config = trigger.config or {}

            for field in required:
                if field not in config or not config[field]:
                    errors.append(ValidationError(
                        code="MISSING_TRIGGER_CONFIG",
                        message=f"Trigger type '{trigger.type.value}' requires '{field}'",
                        path=f"{path}.config.{field}",
                    ))

            # Validate cron expression
            if trigger.type == TriggerType.SCHEDULE:
                cron = config.get("cron_expression", "")
                if cron and not self._is_valid_cron(cron):
                    errors.append(ValidationError(
                        code="INVALID_CRON",
                        message=f"Invalid cron expression: {cron}",
                        path=f"{path}.config.cron_expression",
                    ))

        return errors

    def _validate_steps(self, workflow: Workflow) -> List[ValidationError]:
        """Validate workflow steps."""
        errors = []
        step_ids = set()

        for i, step in enumerate(workflow.steps):
            path = f"steps[{i}]"

            # Check ID
            if not step.id:
                errors.append(ValidationError(
                    code="MISSING_STEP_ID",
                    message="Step must have an ID",
                    path=f"{path}.id",
                    node_id=step.id,
                ))
            elif step.id in step_ids:
                errors.append(ValidationError(
                    code="DUPLICATE_STEP_ID",
                    message=f"Duplicate step ID: {step.id}",
                    path=f"{path}.id",
                    node_id=step.id,
                ))
            else:
                step_ids.add(step.id)

            # Check name
            if not step.name:
                errors.append(ValidationError(
                    code="MISSING_STEP_NAME",
                    message="Step must have a name",
                    path=f"{path}.name",
                    node_id=step.id,
                ))

            # Check action
            if not step.action:
                errors.append(ValidationError(
                    code="MISSING_ACTION",
                    message="Step must have an action",
                    path=f"{path}.action",
                    node_id=step.id,
                ))
            else:
                action_errors = self._validate_action(step.action, f"{path}.action", step.id)
                errors.extend(action_errors)

            # Check approval config
            if step.requires_approval and not step.approval_config:
                errors.append(ValidationError(
                    code="MISSING_APPROVAL_CONFIG",
                    message="Step requires approval but has no approval configuration",
                    path=f"{path}.approval_config",
                    node_id=step.id,
                ))

        return errors

    def _validate_action(
        self,
        action,
        path: str,
        node_id: Optional[str],
    ) -> List[ValidationError]:
        """Validate an action configuration."""
        errors = []

        if not action.type:
            errors.append(ValidationError(
                code="MISSING_ACTION_TYPE",
                message="Action must have a type",
                path=f"{path}.type",
                node_id=node_id,
            ))
            return errors

        # Check required fields
        required = self._action_required_fields.get(action.type, [])
        config = action.config or {}

        for field in required:
            if field not in config or not config[field]:
                errors.append(ValidationError(
                    code="MISSING_ACTION_CONFIG",
                    message=f"Action type '{action.type.value}' requires '{field}'",
                    path=f"{path}.config.{field}",
                    node_id=node_id,
                ))

        # Validate URL format for webhook
        if action.type == ActionType.WEBHOOK:
            url = config.get("url", "")
            if url and not self._is_valid_url(url):
                errors.append(ValidationError(
                    code="INVALID_URL",
                    message=f"Invalid URL: {url}",
                    path=f"{path}.config.url",
                    node_id=node_id,
                ))

        return errors

    def _validate_references(self, workflow: Workflow) -> List[ValidationError]:
        """Validate that all references point to existing steps."""
        errors = []
        step_ids = {step.id for step in workflow.steps}

        for i, step in enumerate(workflow.steps):
            path = f"steps[{i}]"

            # Check depends_on references
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(ValidationError(
                        code="INVALID_DEPENDENCY",
                        message=f"Step '{step.id}' depends on non-existent step '{dep}'",
                        path=f"{path}.depends_on",
                        node_id=step.id,
                    ))

            # Check on_success/on_failure references
            if step.on_success and step.on_success not in step_ids:
                errors.append(ValidationError(
                    code="INVALID_SUCCESS_REFERENCE",
                    message=f"Step '{step.id}' on_success references non-existent step '{step.on_success}'",
                    path=f"{path}.on_success",
                    node_id=step.id,
                ))

            if step.on_failure and step.on_failure not in step_ids:
                errors.append(ValidationError(
                    code="INVALID_FAILURE_REFERENCE",
                    message=f"Step '{step.id}' on_failure references non-existent step '{step.on_failure}'",
                    path=f"{path}.on_failure",
                    node_id=step.id,
                ))

        return errors

    def _detect_cycles(self, workflow: Workflow) -> List[ValidationError]:
        """Detect cycles in workflow dependencies."""
        errors = []

        # Build adjacency list
        adjacency: Dict[str, List[str]] = {}
        for step in workflow.steps:
            adjacency[step.id] = list(step.depends_on)
            if step.on_success:
                adjacency[step.id].append(step.on_success)
            if step.on_failure:
                adjacency[step.id].append(step.on_failure)

        # DFS cycle detection
        visited = set()
        path = set()

        def dfs(node: str) -> Optional[str]:
            visited.add(node)
            path.add(node)

            for neighbor in adjacency.get(node, []):
                if neighbor in path:
                    return neighbor
                if neighbor not in visited:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle

            path.remove(node)
            return None

        for step in workflow.steps:
            if step.id not in visited:
                cycle_node = dfs(step.id)
                if cycle_node:
                    errors.append(ValidationError(
                        code="CYCLE_DETECTED",
                        message=f"Workflow contains a cycle involving step '{cycle_node}'",
                        node_id=cycle_node,
                    ))
                    break  # Report only one cycle

        return errors

    def _find_unreachable_steps(self, workflow: Workflow) -> List[ValidationError]:
        """Find steps that cannot be reached from any trigger."""
        warnings = []

        if not workflow.triggers:
            return warnings

        # Build reverse adjacency (predecessors)
        predecessors: Dict[str, Set[str]] = {step.id: set() for step in workflow.steps}

        for step in workflow.steps:
            for dep in step.depends_on:
                if dep in predecessors:
                    predecessors[step.id].add(dep)

            # Consider on_success/on_failure as edges
            if step.on_success:
                predecessors[step.on_success] = predecessors.get(step.on_success, set())
                # This step leads to on_success

        # Find entry points (steps with no dependencies)
        entry_points = {step.id for step in workflow.steps if not step.depends_on}

        # BFS from entry points
        reachable = set(entry_points)
        queue = list(entry_points)

        while queue:
            current = queue.pop(0)
            step = workflow.get_step(current)
            if not step:
                continue

            # Add successors
            if step.on_success and step.on_success not in reachable:
                reachable.add(step.on_success)
                queue.append(step.on_success)

            if step.on_failure and step.on_failure not in reachable:
                reachable.add(step.on_failure)
                queue.append(step.on_failure)

            # Add steps that depend on this one (they become reachable when this completes)
            for other_step in workflow.steps:
                if current in other_step.depends_on and other_step.id not in reachable:
                    # Check if all dependencies are reachable
                    if all(dep in reachable for dep in other_step.depends_on):
                        reachable.add(other_step.id)
                        queue.append(other_step.id)

        # Find unreachable steps
        all_step_ids = {step.id for step in workflow.steps}
        unreachable = all_step_ids - reachable

        for step_id in unreachable:
            warnings.append(ValidationError(
                code="UNREACHABLE_STEP",
                message=f"Step '{step_id}' is not reachable from any trigger",
                node_id=step_id,
                severity="warning",
            ))

        return warnings

    def _validate_configurations(self, workflow: Workflow) -> List[ValidationError]:
        """Check for incomplete or suspicious configurations."""
        warnings = []

        for i, step in enumerate(workflow.steps):
            path = f"steps[{i}]"

            # Check for empty timeout on potentially long operations
            if step.action and step.action.type in [ActionType.WEBHOOK, ActionType.AGENT, ActionType.LLM]:
                if not step.timeout:
                    warnings.append(ValidationError(
                        code="MISSING_TIMEOUT",
                        message=f"Step '{step.id}' has no timeout configured for a potentially long operation",
                        path=f"{path}.timeout",
                        node_id=step.id,
                        severity="warning",
                    ))

            # Check for missing retry configuration on potentially flaky operations
            if step.action and step.action.type in [ActionType.WEBHOOK]:
                if not step.retry_count or step.retry_count == 0:
                    warnings.append(ValidationError(
                        code="MISSING_RETRY",
                        message=f"Step '{step.id}' has no retry configuration for a webhook action",
                        path=f"{path}.retry_count",
                        node_id=step.id,
                        severity="warning",
                    ))

        return warnings

    def _is_valid_cron(self, expression: str) -> bool:
        """Validate cron expression format."""
        try:
            from croniter import croniter
            croniter(expression)
            return True
        except (ImportError, ValueError):
            # Basic validation: should have 5 or 6 parts
            parts = expression.split()
            return 5 <= len(parts) <= 6

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        # Support template expressions
        if "{{" in url:
            return True

        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


def validate_workflow(workflow: Workflow) -> ValidationResult:
    """Convenience function for workflow validation."""
    validator = WorkflowValidator()
    return validator.validate(workflow)
