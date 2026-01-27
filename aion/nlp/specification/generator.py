"""
AION Specification Generator - Convert intents to formal specifications.

Transforms parsed intents into strongly-typed specification objects
that can be consumed by synthesizers for code generation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import (
    APIEndpointSpec,
    APISpecification,
    AgentSpecification,
    Entity,
    EntityType,
    Intent,
    IntegrationSpecification,
    IntentType,
    ParameterSpec,
    Specification,
    ToolSpecification,
    WorkflowSpecification,
    WorkflowStep,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class SpecificationGenerator:
    """
    Generates formal specifications from parsed intents.

    Each intent type maps to a specific specification builder
    that creates a structured, validated spec object.
    """

    def __init__(self, kernel: AIONKernel):
        self.kernel = kernel

        self._generators = {
            IntentType.CREATE_TOOL: self._generate_tool_spec,
            IntentType.CREATE_WORKFLOW: self._generate_workflow_spec,
            IntentType.CREATE_AGENT: self._generate_agent_spec,
            IntentType.CREATE_API: self._generate_api_spec,
            IntentType.CREATE_INTEGRATION: self._generate_integration_spec,
            IntentType.CREATE_FUNCTION: self._generate_tool_spec,
        }

    async def generate(self, intent: Intent) -> Specification:
        """
        Generate a specification from an intent.

        Args:
            intent: Parsed intent with entities and parameters

        Returns:
            Typed specification object

        Raises:
            ValueError: If no generator exists for the intent type
        """
        generator = self._generators.get(intent.type)
        if not generator:
            raise ValueError(f"No specification generator for intent type: {intent.type.value}")

        spec = await generator(intent)

        # LLM enhancement pass
        spec = await self._enhance_with_llm(spec, intent)

        logger.info(
            "Specification generated",
            type=intent.type.value,
            name=getattr(spec, "name", "unknown"),
        )

        return spec

    # =========================================================================
    # Tool Specification
    # =========================================================================

    async def _generate_tool_spec(self, intent: Intent) -> ToolSpecification:
        """Generate a tool specification."""
        params = intent.parameters

        # Build parameter specs
        parameters = self._build_parameters(params.get("inputs", []))

        # Detect API integration
        api_endpoint = None
        api_method = "GET"
        api_headers: Dict[str, str] = {}
        auth_required = False
        auth_type = None

        raw_lower = intent.raw_input.lower()

        # Extract API endpoint from entities or text
        endpoint_entity = intent.get_entity(EntityType.API_ENDPOINT)
        if endpoint_entity:
            api_endpoint = endpoint_entity.value
        elif "api" in raw_lower or "fetch" in raw_lower or "http" in raw_lower:
            url_match = re.search(r'https?://[^\s]+', intent.raw_input)
            if url_match:
                api_endpoint = url_match.group()

        # Detect HTTP method
        method_entity = intent.get_entity(EntityType.API_METHOD)
        if method_entity:
            api_method = method_entity.value.upper()
        elif "post" in raw_lower:
            api_method = "POST"
        elif "put" in raw_lower:
            api_method = "PUT"
        elif "delete" in raw_lower and "api" in raw_lower:
            api_method = "DELETE"

        # Auth detection
        auth_entity = intent.get_entity(EntityType.AUTH_TYPE)
        if auth_entity:
            auth_required = True
            auth_type = auth_entity.value
        elif any(w in raw_lower for w in ["auth", "token", "api key", "credentials"]):
            auth_required = True
            auth_type = "bearer"

        return ToolSpecification(
            name=self._clean_name(params.get("name", "custom_tool")),
            description=params.get("description", intent.raw_input),
            parameters=parameters,
            return_type=params.get("return_type", "Any"),
            return_description=params.get("return_description", ""),
            implementation_notes=params.get("notes", ""),
            dependencies=params.get("dependencies", []),
            api_endpoint=api_endpoint,
            api_method=api_method,
            api_headers=api_headers,
            auth_required=auth_required,
            auth_type=auth_type,
        )

    # =========================================================================
    # Workflow Specification
    # =========================================================================

    async def _generate_workflow_spec(self, intent: Intent) -> WorkflowSpecification:
        """Generate a workflow specification."""
        params = intent.parameters

        # Determine trigger
        trigger_type, trigger_config = self._build_trigger(intent)

        # Build steps
        steps = self._build_workflow_steps(params, intent)

        # Build inputs
        inputs = self._build_parameters(params.get("inputs", []))

        return WorkflowSpecification(
            name=self._clean_name(params.get("name", "custom_workflow")),
            description=params.get("description", intent.raw_input),
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            steps=steps,
            on_error=params.get("on_error", "stop"),
            max_retries=params.get("max_retries", 3),
            inputs=inputs,
        )

    # =========================================================================
    # Agent Specification
    # =========================================================================

    async def _generate_agent_spec(self, intent: Intent) -> AgentSpecification:
        """Generate an agent specification."""
        params = intent.parameters
        raw_lower = intent.raw_input.lower()

        # Extract personality traits
        traits = self._extract_traits(raw_lower)

        # Build system prompt
        description = params.get("description", intent.raw_input)
        goal = params.get("goal", description)
        constraints = params.get("constraints", [])

        system_prompt = self._build_system_prompt(description, traits, constraints)

        return AgentSpecification(
            name=self._clean_name(params.get("name", "custom_agent")),
            description=description,
            system_prompt=system_prompt,
            personality_traits=traits,
            primary_goal=goal,
            sub_goals=params.get("sub_goals", []),
            constraints=constraints,
            allowed_tools=params.get("tools", []),
            allowed_actions=params.get("actions", []),
            success_criteria=params.get("success_criteria", []),
        )

    # =========================================================================
    # API Specification
    # =========================================================================

    async def _generate_api_spec(self, intent: Intent) -> APISpecification:
        """Generate an API specification."""
        params = intent.parameters

        # Build endpoints
        endpoints = self._build_api_endpoints(params, intent)

        # Auth
        auth_entity = intent.get_entity(EntityType.AUTH_TYPE)
        auth_type = auth_entity.value if auth_entity else params.get("auth_type")

        return APISpecification(
            name=self._clean_name(params.get("name", "custom_api")),
            description=params.get("description", intent.raw_input),
            endpoints=endpoints,
            auth_type=auth_type,
        )

    # =========================================================================
    # Integration Specification
    # =========================================================================

    async def _generate_integration_spec(self, intent: Intent) -> IntegrationSpecification:
        """Generate an integration specification."""
        params = intent.parameters

        # Extract source and target
        source = ""
        target = ""

        source_entity = intent.get_entity(EntityType.DATA_SOURCE)
        if source_entity:
            source = source_entity.value

        target_entity = intent.get_entity(EntityType.DATA_TARGET)
        if target_entity:
            target = target_entity.value

        name = params.get("name", "")
        if not name and source and target:
            name = f"{source}_to_{target}_integration"
        elif not name:
            name = "custom_integration"

        return IntegrationSpecification(
            name=self._clean_name(name),
            description=params.get("description", intent.raw_input),
            source_system=source,
            source_config=params.get("source_config", {}),
            target_system=target,
            target_config=params.get("target_config", {}),
            field_mapping=params.get("mapping", []),
            sync_mode=params.get("sync_mode", "incremental"),
        )

    # =========================================================================
    # LLM Enhancement
    # =========================================================================

    async def _enhance_with_llm(self, spec: Specification, intent: Intent) -> Specification:
        """Use LLM to fill gaps and improve the specification."""
        spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else {}

        prompt = f"""Enhance this system specification. Fill in missing details with sensible defaults.

Original request: "{intent.raw_input}"

Current specification:
{spec_dict}

Rules:
- Only add information that is clearly implied by the request
- Use sensible defaults for missing values
- Do not change the name or description
- Focus on parameters, error handling, and edge cases

Respond with ONLY valid JSON containing the enhanced fields:
{{
    "enhanced_parameters": [
        {{"name": "<name>", "type": "<type>", "description": "<desc>", "required": <bool>}}
    ],
    "implementation_hints": "<hints>",
    "error_handling": "<strategy>",
    "suggested_dependencies": ["<dep>"]
}}"""

        try:
            response = await self.kernel.llm.complete(
                [{"role": "user", "content": prompt}]
            )
            content = response.content if hasattr(response, "content") else str(response)

            import json
            data = json.loads(content) if content.strip().startswith("{") else {}

            # Merge enhanced parameters (only if spec has empty parameters)
            if data.get("enhanced_parameters") and hasattr(spec, "parameters"):
                if not spec.parameters:
                    for p in data["enhanced_parameters"]:
                        if isinstance(spec.parameters, list):
                            spec.parameters.append(ParameterSpec(
                                name=p.get("name", "param"),
                                type=p.get("type", "string"),
                                description=p.get("description", ""),
                                required=p.get("required", True),
                            ))

            if data.get("implementation_hints") and hasattr(spec, "implementation_notes"):
                if not spec.implementation_notes:
                    spec.implementation_notes = data["implementation_hints"]

        except Exception as e:
            logger.debug("LLM spec enhancement skipped", error=str(e))

        return spec

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_parameters(self, inputs: List[Any]) -> List[ParameterSpec]:
        """Build ParameterSpec list from raw parameter data."""
        parameters: List[ParameterSpec] = []
        for p in inputs:
            if isinstance(p, dict):
                parameters.append(ParameterSpec(
                    name=p.get("name", "param"),
                    type=p.get("type", "string"),
                    description=p.get("description", ""),
                    required=p.get("required", True),
                    default=p.get("default"),
                ))
            elif isinstance(p, str):
                parameters.append(ParameterSpec(name=p, type="string"))
        return parameters

    def _build_trigger(self, intent: Intent) -> tuple[str, Dict[str, Any]]:
        """Build trigger type and config from intent."""
        params = intent.parameters
        triggers = params.get("triggers", [])
        trigger_entities = intent.get_entities(EntityType.TRIGGER)
        schedule_entities = intent.get_entities(EntityType.SCHEDULE)

        # Check schedule entities first
        if schedule_entities:
            schedule_text = " ".join(e.value for e in schedule_entities)
            return "schedule", self._parse_schedule(schedule_text)

        # Check trigger entities
        if trigger_entities:
            trigger_text = trigger_entities[0].value.lower()
            if any(w in trigger_text for w in ["every", "daily", "hourly", "weekly"]):
                return "schedule", self._parse_schedule(trigger_text)
            elif "webhook" in trigger_text:
                return "webhook", {"path": f"/webhook/{intent.parameters.get('name', 'hook')}"}
            else:
                return "event", {"condition": trigger_text}

        # Check from params
        if triggers:
            trigger = triggers[0] if isinstance(triggers[0], str) else str(triggers[0])
            trigger_lower = trigger.lower()
            if any(w in trigger_lower for w in ["every", "daily", "hourly", "weekly"]):
                return "schedule", self._parse_schedule(trigger_lower)
            elif "webhook" in trigger_lower:
                return "webhook", {}
            else:
                return "event", {"condition": trigger}

        return "manual", {}

    def _parse_schedule(self, text: str) -> Dict[str, Any]:
        """Parse schedule configuration from text."""
        config: Dict[str, Any] = {"type": "cron", "expression": "0 * * * *"}

        if "every day" in text or "daily" in text:
            config["expression"] = "0 0 * * *"
        elif "every hour" in text or "hourly" in text:
            config["expression"] = "0 * * * *"
        elif "every minute" in text:
            config["expression"] = "* * * * *"
        elif "every week" in text or "weekly" in text:
            config["expression"] = "0 0 * * 0"
        elif "every month" in text or "monthly" in text:
            config["expression"] = "0 0 1 * *"

        # Extract specific time
        time_match = re.search(r'(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text, re.I)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            ampm = time_match.group(3)
            if ampm and ampm.lower() == "pm" and hour < 12:
                hour += 12
            elif ampm and ampm.lower() == "am" and hour == 12:
                hour = 0
            config["expression"] = f"{minute} {hour} * * *"

        return config

    def _build_workflow_steps(
        self,
        params: Dict[str, Any],
        intent: Intent,
    ) -> List[WorkflowStep]:
        """Build workflow steps from parameters and entities."""
        steps: List[WorkflowStep] = []
        raw_steps = params.get("inputs", [])

        for i, step_data in enumerate(raw_steps):
            if isinstance(step_data, dict):
                steps.append(WorkflowStep(
                    id=f"step_{i + 1}",
                    name=step_data.get("name", f"Step {i + 1}"),
                    action=step_data.get("action", step_data.get("description", "execute")),
                    params=step_data.get("params", {}),
                ))
            elif isinstance(step_data, str):
                steps.append(WorkflowStep(
                    id=f"step_{i + 1}",
                    name=f"Step {i + 1}",
                    action=step_data,
                ))

        # If no steps but we have action entities, use those
        if not steps:
            action_entities = intent.get_entities(EntityType.ACTION)
            for i, entity in enumerate(action_entities):
                steps.append(WorkflowStep(
                    id=f"step_{i + 1}",
                    name=entity.value,
                    action=entity.value,
                ))

        return steps

    def _build_api_endpoints(
        self,
        params: Dict[str, Any],
        intent: Intent,
    ) -> List[APIEndpointSpec]:
        """Build API endpoint specifications."""
        endpoints: List[APIEndpointSpec] = []

        raw_endpoints = params.get("outputs", [])
        for ep_data in raw_endpoints:
            if isinstance(ep_data, dict):
                endpoints.append(APIEndpointSpec(
                    path=ep_data.get("path", "/resource"),
                    method=ep_data.get("method", "GET"),
                    description=ep_data.get("description", ""),
                ))

        # Generate default CRUD if no endpoints specified
        if not endpoints:
            resource = self._clean_name(params.get("name", "resource"))
            endpoints = [
                APIEndpointSpec(path=f"/{resource}", method="GET", description=f"List all {resource}"),
                APIEndpointSpec(path=f"/{resource}", method="POST", description=f"Create a {resource}"),
                APIEndpointSpec(path=f"/{resource}/{{id}}", method="GET", description=f"Get a {resource} by ID"),
                APIEndpointSpec(path=f"/{resource}/{{id}}", method="PUT", description=f"Update a {resource}"),
                APIEndpointSpec(path=f"/{resource}/{{id}}", method="DELETE", description=f"Delete a {resource}"),
            ]

        return endpoints

    def _extract_traits(self, text: str) -> List[str]:
        """Extract personality traits from text."""
        trait_keywords = {
            "helpful": "helpful", "friendly": "friendly",
            "professional": "professional", "concise": "concise",
            "detailed": "thorough", "creative": "creative",
            "analytical": "analytical", "proactive": "proactive",
            "careful": "careful", "fast": "efficient",
            "thorough": "thorough", "strict": "strict",
            "patient": "patient", "technical": "technical",
        }
        traits = []
        for keyword, trait in trait_keywords.items():
            if keyword in text:
                traits.append(trait)
        return traits or ["helpful", "professional"]

    def _build_system_prompt(
        self,
        description: str,
        traits: List[str],
        constraints: List[str],
    ) -> str:
        """Build a system prompt for an agent."""
        parts = [f"You are an AI agent with the following purpose:\n{description}"]

        if traits:
            parts.append(f"\nYour personality traits: {', '.join(traits)}")

        if constraints:
            parts.append("\nConstraints:")
            for c in constraints:
                parts.append(f"- {c}")

        parts.append("\nFollow your goals and constraints carefully.")
        return "\n".join(parts)

    def _clean_name(self, name: str) -> str:
        """Clean a name for use as an identifier."""
        name = re.sub(r'[^a-zA-Z0-9_\s]', '', name)
        name = re.sub(r'\s+', '_', name.strip())
        name = name.lower()
        # Ensure valid Python identifier
        if name and name[0].isdigit():
            name = f"sys_{name}"
        return name or "unnamed"
