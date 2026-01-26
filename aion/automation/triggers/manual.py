"""
AION Manual Trigger Handler

User-initiated manual triggers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from aion.automation.types import Trigger, TriggerType
from aion.automation.triggers.manager import BaseTriggerHandler

logger = structlog.get_logger(__name__)


class ManualTriggerHandler(BaseTriggerHandler):
    """
    Handler for manual triggers.

    Features:
    - User-initiated execution
    - Input validation
    - Access control integration
    """

    async def register(self, trigger: Trigger) -> None:
        """Register a manual trigger."""
        logger.info(
            "manual_trigger_registered",
            trigger_id=trigger.id,
            workflow_id=trigger.workflow_id,
        )

    async def unregister(self, trigger: Trigger) -> None:
        """Unregister a manual trigger."""
        logger.info("manual_trigger_unregistered", trigger_id=trigger.id)

    async def execute(
        self,
        workflow_id: str,
        inputs: Dict[str, Any] = None,
        initiated_by: str = None,
        dry_run: bool = False,
    ) -> Optional[str]:
        """
        Manually execute a workflow.

        Args:
            workflow_id: Workflow to execute
            inputs: Input data
            initiated_by: User ID who initiated
            dry_run: If True, validate without executing

        Returns:
            Execution ID (or None for dry run)
        """
        if dry_run:
            # Validate inputs against schema (would be implemented)
            logger.info(
                "manual_dry_run",
                workflow_id=workflow_id,
                inputs=inputs,
            )
            return None

        execution_id = await self.manager.handle_manual(
            workflow_id=workflow_id,
            inputs=inputs,
            initiated_by=initiated_by,
        )

        return execution_id

    async def validate_inputs(
        self,
        workflow_id: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate inputs against workflow schema.

        Returns:
            Validation result with errors if any
        """
        # Get workflow
        workflow = await self.manager.engine.get_workflow(workflow_id)
        if not workflow:
            return {
                "valid": False,
                "errors": [f"Workflow not found: {workflow_id}"],
            }

        errors = []

        # Check required inputs from schema
        schema = workflow.input_schema
        if schema and "required" in schema:
            for required_field in schema["required"]:
                if required_field not in (inputs or {}):
                    errors.append(f"Missing required input: {required_field}")

        # Check input types (basic validation)
        if schema and "properties" in schema:
            for field, spec in schema["properties"].items():
                if field in (inputs or {}):
                    value = inputs[field]
                    expected_type = spec.get("type")

                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Field '{field}' must be a string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        errors.append(f"Field '{field}' must be a number")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(f"Field '{field}' must be a boolean")
                    elif expected_type == "array" and not isinstance(value, list):
                        errors.append(f"Field '{field}' must be an array")
                    elif expected_type == "object" and not isinstance(value, dict):
                        errors.append(f"Field '{field}' must be an object")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    def can_execute(
        self,
        workflow_id: str,
        user_id: str,
    ) -> bool:
        """
        Check if user can manually execute a workflow.

        This would integrate with the security system.
        """
        # Placeholder - would integrate with aion.security
        return True
