"""
AION Workflow Synthesizer - Generate workflow configurations.

Creates workflow definitions with:
- DAG-based step execution
- Trigger configuration
- Error handling and retries
- Step handler implementations
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import structlog

from aion.nlp.synthesis.base import BaseSynthesizer
from aion.nlp.types import (
    GeneratedCode,
    SpecificationType,
    WorkflowSpecification,
    WorkflowStep,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


_TRIGGER_CLASS_MAP = {
    "schedule": "ScheduleTrigger",
    "event": "EventTrigger",
    "webhook": "WebhookTrigger",
    "manual": "ManualTrigger",
}


class WorkflowSynthesizer(BaseSynthesizer):
    """
    Synthesizes workflow definitions from WorkflowSpecification.

    Generates:
    - Workflow configuration with trigger, steps, error handling
    - Step handler async functions
    - Registration code
    """

    async def synthesize(self, spec: WorkflowSpecification) -> GeneratedCode:
        """Generate workflow code from specification."""
        # Generate workflow definition
        workflow_def = self._generate_workflow_definition(spec)

        # Generate step handlers
        step_handlers = await self._generate_step_handlers(spec)

        # Generate registration
        registration = self._generate_registration(spec)

        trigger_class = _TRIGGER_CLASS_MAP.get(spec.trigger_type, "ManualTrigger")

        code = f'''"""
Workflow: {spec.name}
Description: {spec.description}
Trigger: {spec.trigger_type}
Steps: {len(spec.steps)}
"""

from typing import Any, Dict
from aion.workflows import Workflow, Step, Trigger
from aion.workflows.triggers import {trigger_class}

# Step Handlers
{step_handlers}

# Workflow Definition
{workflow_def}

# Registration
{registration}
'''

        return GeneratedCode(
            language="python",
            code=code.strip(),
            filename=f"workflow_{spec.name}.py",
            spec_type=SpecificationType.WORKFLOW,
            imports=[
                "from typing import Any, Dict",
                "from aion.workflows import Workflow, Step, Trigger",
            ],
            docstring=spec.description,
        )

    def _generate_workflow_definition(self, spec: WorkflowSpecification) -> str:
        """Generate the workflow object definition."""
        trigger_code = self._generate_trigger_code(spec)
        steps_code = self._generate_steps_code(spec)

        return f"""
workflow = Workflow(
    name="{spec.name}",
    description="{spec.description}",
    trigger={trigger_code},
    steps=[
{steps_code}
    ],
    on_error="{spec.on_error}",
    max_retries={spec.max_retries},
    max_parallel_steps={spec.max_parallel_steps},
    timeout_seconds={spec.timeout_seconds},
)
"""

    def _generate_trigger_code(self, spec: WorkflowSpecification) -> str:
        """Generate trigger instantiation code."""
        if spec.trigger_type == "schedule":
            cron = spec.trigger_config.get("expression", "0 * * * *")
            return f'ScheduleTrigger(cron="{cron}")'
        elif spec.trigger_type == "event":
            condition = spec.trigger_config.get("condition", "")
            return f'EventTrigger(condition="{condition}")'
        elif spec.trigger_type == "webhook":
            path = spec.trigger_config.get("path", f"/webhook/{spec.name}")
            return f'WebhookTrigger(path="{path}")'
        else:
            return "ManualTrigger()"

    def _generate_steps_code(self, spec: WorkflowSpecification) -> str:
        """Generate step definitions."""
        step_lines: List[str] = []
        for step in spec.steps:
            deps = repr(step.depends_on) if step.depends_on else "[]"
            step_lines.append(f"""        Step(
            id="{step.id}",
            name="{step.name}",
            action="{step.action}",
            params={step.params or {}},
            depends_on={deps},
            handler=handle_{step.id},
            on_error="{step.on_error}",
            timeout_seconds={step.timeout_seconds},
        )""")
        return ",\n".join(step_lines)

    async def _generate_step_handlers(self, spec: WorkflowSpecification) -> str:
        """Generate handler functions for each step."""
        handlers: List[str] = []

        for step in spec.steps:
            handler = await self._generate_single_handler(step, spec)
            handlers.append(handler)

        return "\n\n".join(handlers)

    async def _generate_single_handler(
        self,
        step: WorkflowStep,
        spec: WorkflowSpecification,
    ) -> str:
        """Generate a single step handler."""
        prompt = f"""Generate a Python async function for this workflow step:

Step name: {step.name}
Action: {step.action}
Parameters: {step.params}
Workflow: {spec.description}

Requirements:
- Function signature: async def handle_{step.id}(context: Dict[str, Any]) -> Any
- Accept a context dict with data from previous steps
- Return results to pass to next steps
- Include error handling
- Be practical and concise

Generate ONLY the function (with def statement):"""

        try:
            code = await self._llm_generate(prompt)
            # Ensure correct function name
            if f"handle_{step.id}" not in code:
                code = f"""async def handle_{step.id}(context: Dict[str, Any]) -> Any:
    \"\"\"Handle step: {step.name} - {step.action}\"\"\"
{self._indent(code)}"""
            return code
        except Exception as e:
            logger.warning("LLM step generation failed", step=step.id, error=str(e))
            return f"""async def handle_{step.id}(context: Dict[str, Any]) -> Any:
    \"\"\"Handle step: {step.name} - {step.action}\"\"\"
    # TODO: Implement {step.action}
    raise NotImplementedError("Step handler not implemented")
"""

    def _generate_registration(self, spec: WorkflowSpecification) -> str:
        """Generate workflow registration code."""
        return f"""
def register_workflow():
    \"\"\"Register the {spec.name} workflow with AION.\"\"\"
    return workflow
"""
