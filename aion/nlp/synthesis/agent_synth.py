"""
AION Agent Synthesizer - Generate agent configurations and logic.

Creates autonomous agent definitions with:
- System prompts and personality
- Goal hierarchies
- Tool access control
- Memory configuration
- Constraint enforcement
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import structlog

from aion.nlp.synthesis.base import BaseSynthesizer
from aion.nlp.types import (
    AgentSpecification,
    GeneratedCode,
    SpecificationType,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class AgentSynthesizer(BaseSynthesizer):
    """
    Synthesizes agent configurations from AgentSpecification.

    Generates:
    - Agent configuration dataclass
    - System prompt with personality
    - Goal-driven behavior logic
    - Tool access definitions
    - Constraint enforcement
    """

    async def synthesize(self, spec: AgentSpecification) -> GeneratedCode:
        """Generate agent code from specification."""
        # Generate agent configuration
        config_code = self._generate_config(spec)

        # Generate behavior logic
        behavior_code = await self._generate_behavior(spec)

        # Generate constraint checks
        constraint_code = self._generate_constraints(spec)

        code = f'''"""
Agent: {spec.name}
Description: {spec.description}
Primary Goal: {spec.primary_goal}
Traits: {", ".join(spec.personality_traits)}
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Agent Configuration
{config_code}

# Constraint Checks
{constraint_code}

# Behavior Logic
{behavior_code}

# Registration
def register_agent():
    """Register the {spec.name} agent with AION."""
    return agent_config
'''

        return GeneratedCode(
            language="python",
            code=code.strip(),
            filename=f"agent_{spec.name}.py",
            spec_type=SpecificationType.AGENT,
            imports=[
                "from dataclasses import dataclass, field",
                "from typing import Any, Dict, List, Optional",
            ],
            docstring=spec.description,
        )

    def _generate_config(self, spec: AgentSpecification) -> str:
        """Generate agent configuration."""
        tools_str = repr(spec.allowed_tools)
        constraints_str = repr(spec.constraints)
        traits_str = repr(spec.personality_traits)

        return f'''
@dataclass
class {self._class_name(spec.name)}Config:
    """Configuration for {spec.name} agent."""

    name: str = "{spec.name}"
    description: str = """{spec.description}"""

    # System prompt
    system_prompt: str = """{spec.system_prompt}"""

    # Goals
    primary_goal: str = """{spec.primary_goal}"""
    sub_goals: List[str] = field(default_factory=lambda: {repr(spec.sub_goals)})
    success_criteria: List[str] = field(default_factory=lambda: {repr(spec.success_criteria)})

    # Personality
    personality_traits: List[str] = field(default_factory=lambda: {traits_str})
    communication_style: str = "{spec.communication_style}"

    # Capabilities
    allowed_tools: List[str] = field(default_factory=lambda: {tools_str})
    allowed_actions: List[str] = field(default_factory=lambda: {repr(spec.allowed_actions)})

    # Constraints
    constraints: List[str] = field(default_factory=lambda: {constraints_str})
    forbidden_actions: List[str] = field(default_factory=lambda: {repr(spec.forbidden_actions)})

    # Resource limits
    max_iterations: int = {spec.max_iterations}
    timeout_seconds: float = {spec.timeout_seconds}
    max_tokens_per_turn: int = {spec.max_tokens_per_turn}
    max_tool_calls: int = {spec.max_tool_calls}

    # Memory
    memory_enabled: bool = {spec.memory_enabled}
    memory_types: List[str] = field(default_factory=lambda: {repr(spec.memory_types)})

    # Collaboration
    can_delegate: bool = {spec.can_delegate}
    can_collaborate: bool = {spec.can_collaborate}

    # Learning
    learn_from_feedback: bool = {spec.learn_from_feedback}


agent_config = {self._class_name(spec.name)}Config()
'''

    async def _generate_behavior(self, spec: AgentSpecification) -> str:
        """Generate agent behavior logic."""
        prompt = f"""Generate Python async functions for an AI agent with these characteristics:

Agent: {spec.name}
Description: {spec.description}
Primary Goal: {spec.primary_goal}
Sub-goals: {spec.sub_goals}
Allowed Tools: {spec.allowed_tools}
Constraints: {spec.constraints}

Generate two functions:
1. `async def evaluate_situation(context: Dict[str, Any]) -> Dict[str, Any]`
   - Analyzes the current situation against goals
   - Returns analysis with recommended actions

2. `async def execute_action(action: str, context: Dict[str, Any]) -> Dict[str, Any]`
   - Executes a specific action
   - Returns the result

Requirements:
- Include docstrings
- Handle errors gracefully
- Respect constraints
- Be practical and concise"""

        try:
            code = await self._llm_generate(prompt)
            return code
        except Exception as e:
            logger.warning("LLM agent behavior generation failed", error=str(e))
            return f'''
async def evaluate_situation(context: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate current situation against agent goals."""
    return {{
        "goal": "{spec.primary_goal}",
        "status": "pending",
        "recommended_actions": [],
    }}


async def execute_action(action: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an agent action."""
    # TODO: Implement agent behavior for {spec.name}
    return {{
        "action": action,
        "status": "not_implemented",
    }}
'''

    def _generate_constraints(self, spec: AgentSpecification) -> str:
        """Generate constraint checking code."""
        if not spec.constraints and not spec.forbidden_actions:
            return """
def check_constraints(action: str, context: Dict[str, Any]) -> bool:
    \"\"\"Check if an action is allowed.\"\"\"
    return True
"""

        checks: List[str] = []

        for forbidden in spec.forbidden_actions:
            checks.append(f'    if action == "{forbidden}":\n        return False')

        constraint_list = repr(spec.constraints)

        return f"""
CONSTRAINTS = {constraint_list}
FORBIDDEN_ACTIONS = {repr(spec.forbidden_actions)}


def check_constraints(action: str, context: Dict[str, Any]) -> bool:
    \"\"\"Check if an action is allowed given agent constraints.\"\"\"
    # Check forbidden actions
    if action in FORBIDDEN_ACTIONS:
        return False

{chr(10).join(checks)}

    return True
"""

    def _class_name(self, name: str) -> str:
        """Convert snake_case name to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))
