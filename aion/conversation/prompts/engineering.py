"""
AION SOTA Prompt Engineering

Advanced prompt engineering featuring:
- Chain-of-Thought (CoT) prompting for complex reasoning
- ReAct (Reasoning + Acting) patterns for tool use
- Self-consistency prompting for improved accuracy
- Constitutional AI patterns for safety
- Dynamic prompt optimization based on task type
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

class PromptType(str, Enum):
    """Types of prompts."""
    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    SELF_CONSISTENCY = "self_consistency"
    CONSTITUTIONAL = "constitutional"
    FEW_SHOT = "few_shot"
    TREE_OF_THOUGHT = "tree_of_thought"


@dataclass
class PromptTemplate:
    """A reusable prompt template."""
    name: str
    template: str
    prompt_type: PromptType = PromptType.STANDARD
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables."""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result

    def with_examples(self, examples: List[Dict[str, str]]) -> str:
        """Add few-shot examples to the template."""
        example_text = "\n\n".join(
            f"Example {i+1}:\nInput: {ex.get('input', '')}\nOutput: {ex.get('output', '')}"
            for i, ex in enumerate(examples)
        )
        return f"{example_text}\n\n{self.template}"


# =============================================================================
# Chain-of-Thought Prompting
# =============================================================================

class ChainOfThoughtPrompt:
    """
    Chain-of-Thought (CoT) prompting for complex reasoning tasks.

    CoT prompting encourages the model to "think step by step" which
    significantly improves performance on reasoning tasks like math,
    logic, and multi-step problems.
    """

    STANDARD_COT_PREFIX = """Let me think through this step by step:

"""

    ZERO_SHOT_COT_SUFFIX = """

Let's approach this systematically:
1. First, I'll identify the key information and constraints
2. Then, I'll break down the problem into smaller steps
3. Finally, I'll solve each step and combine the results

"""

    STRUCTURED_COT_TEMPLATE = """## Problem Analysis

**Given Information:**
{given}

**Goal:**
{goal}

**Constraints:**
{constraints}

## Step-by-Step Reasoning

{reasoning_steps}

## Conclusion

{conclusion}"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def wrap_query(
        self,
        query: str,
        cot_type: str = "zero_shot",
    ) -> str:
        """
        Wrap a query with CoT prompting.

        Args:
            query: The original query
            cot_type: Type of CoT ("zero_shot", "few_shot", "structured")

        Returns:
            Query with CoT prompting
        """
        if cot_type == "zero_shot":
            return f"{query}\n\nLet's think step by step."

        elif cot_type == "structured":
            return f"""{query}

Please solve this by:
1. Stating what information is given
2. Identifying what we need to find
3. Listing any constraints or assumptions
4. Working through the solution step by step
5. Verifying the answer

Show your complete reasoning process."""

        else:  # Default
            return f"{query}{self.ZERO_SHOT_COT_SUFFIX}"

    def create_math_cot(self, problem: str) -> str:
        """Create CoT prompt for mathematical problems."""
        return f"""Mathematical Problem: {problem}

Let me solve this step by step:

Step 1: Understand what we're asked to find
[Identify the question]

Step 2: List the given information
[Extract relevant numbers and relationships]

Step 3: Choose an approach
[Select appropriate mathematical method]

Step 4: Execute the solution
[Show all calculations]

Step 5: Verify the answer
[Check if the answer makes sense]

Final Answer: """

    def create_logic_cot(self, problem: str) -> str:
        """Create CoT prompt for logical reasoning."""
        return f"""Logical Problem: {problem}

Let me reason through this carefully:

Premises:
[List all given statements]

Analysis:
[Examine each premise and identify logical relationships]

Deduction:
[Apply logical rules step by step]

Conclusion:
[State the logical conclusion with justification]

"""

    def create_code_cot(self, task: str) -> str:
        """Create CoT prompt for coding tasks."""
        return f"""Coding Task: {task}

Let me approach this systematically:

1. **Understanding the Requirements**
   - What input do we receive?
   - What output do we need to produce?
   - What are the constraints?

2. **Algorithm Design**
   - What's the high-level approach?
   - What data structures should we use?
   - What's the time/space complexity?

3. **Implementation Plan**
   - Break down into functions/methods
   - Handle edge cases
   - Add error handling

4. **Code**
```python
# Implementation here
```

5. **Testing**
   - Test with example inputs
   - Test edge cases
   - Verify correctness

"""


# =============================================================================
# ReAct Prompting
# =============================================================================

class ReActPrompt:
    """
    ReAct (Reasoning + Acting) prompting for tool-using agents.

    ReAct interleaves reasoning traces with actions, allowing the model
    to plan, execute tools, observe results, and adjust its approach.
    """

    REACT_SYSTEM_PROMPT = """You are an AI assistant that uses a structured approach to solve problems by reasoning and taking actions.

For each step, you will:
1. **Thought**: Reason about what to do next based on the current situation
2. **Action**: Decide on an action to take (use a tool or provide final answer)
3. **Observation**: Receive the result of your action

Continue this loop until you have enough information to provide a final answer.

Format your responses as:
Thought: [Your reasoning about what to do]
Action: [tool_name] with input: [tool_input]
or
Action: Final Answer: [your final answer]

Available tools:
{tools}

Remember:
- Think before acting
- Use tools when needed for information you don't have
- Synthesize observations to form your answer
- Be prepared to adjust your approach based on observations
"""

    REACT_STEP_TEMPLATE = """Thought: {thought}
Action: {action}
Observation: {observation}
"""

    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None):
        self.tools = tools or []

    def create_system_prompt(self, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Create ReAct system prompt with available tools."""
        tool_list = tools or self.tools

        tool_descriptions = []
        for tool in tool_list:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            params = tool.get("parameters", {})
            tool_descriptions.append(
                f"- **{name}**: {description}\n  Parameters: {json.dumps(params, indent=2)}"
            )

        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."

        return self.REACT_SYSTEM_PROMPT.format(tools=tools_text)

    def format_step(
        self,
        thought: str,
        action: str,
        observation: str,
    ) -> str:
        """Format a single ReAct step."""
        return self.REACT_STEP_TEMPLATE.format(
            thought=thought,
            action=action,
            observation=observation,
        )

    def create_react_prompt(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Create a ReAct prompt for a query.

        Args:
            query: The user's query
            history: Previous ReAct steps (thought/action/observation)

        Returns:
            Formatted ReAct prompt
        """
        prompt_parts = [f"Question: {query}\n"]

        if history:
            for i, step in enumerate(history, 1):
                prompt_parts.append(f"\nStep {i}:")
                prompt_parts.append(self.format_step(
                    step.get("thought", ""),
                    step.get("action", ""),
                    step.get("observation", ""),
                ))

        prompt_parts.append("\nNow, what is your next thought and action?")

        return "\n".join(prompt_parts)

    def parse_react_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a ReAct response to extract thought, action, and inputs.

        Returns:
            Dict with 'thought', 'action', 'action_input', 'is_final'
        """
        result = {
            "thought": "",
            "action": "",
            "action_input": "",
            "is_final": False,
        }

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response, re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()

            # Check for final answer
            if action_text.lower().startswith("final answer:"):
                result["is_final"] = True
                result["action"] = "final_answer"
                result["action_input"] = action_text[13:].strip()
            else:
                # Parse tool action
                tool_match = re.match(r'(\w+)\s*(?:with input:|:)?\s*(.+)?', action_text, re.DOTALL)
                if tool_match:
                    result["action"] = tool_match.group(1)
                    result["action_input"] = tool_match.group(2).strip() if tool_match.group(2) else ""

        return result


# =============================================================================
# Constitutional AI Prompting
# =============================================================================

class ConstitutionalPrompt:
    """
    Constitutional AI prompting for safety and alignment.

    Uses a set of principles (constitution) to guide and critique
    model outputs, reducing harmful or biased responses.
    """

    DEFAULT_PRINCIPLES = [
        {
            "name": "helpfulness",
            "description": "The response should be helpful and address the user's needs.",
            "critique": "Is this response helpful? Does it address what the user asked?",
        },
        {
            "name": "harmlessness",
            "description": "The response should not cause harm or promote harmful actions.",
            "critique": "Could this response cause harm? Does it promote dangerous activities?",
        },
        {
            "name": "honesty",
            "description": "The response should be truthful and acknowledge uncertainty.",
            "critique": "Is this response honest? Does it acknowledge what it doesn't know?",
        },
        {
            "name": "privacy",
            "description": "The response should respect user privacy and not request sensitive information.",
            "critique": "Does this response respect privacy? Does it avoid requesting sensitive data?",
        },
        {
            "name": "fairness",
            "description": "The response should be unbiased and treat all groups fairly.",
            "critique": "Is this response fair and unbiased? Does it avoid stereotypes?",
        },
    ]

    def __init__(self, principles: Optional[List[Dict[str, str]]] = None):
        self.principles = principles or self.DEFAULT_PRINCIPLES

    def create_critique_prompt(
        self,
        response: str,
        principles: Optional[List[str]] = None,
    ) -> str:
        """
        Create a prompt to critique a response against principles.

        Args:
            response: The response to critique
            principles: Specific principles to check (or all if None)

        Returns:
            Critique prompt
        """
        if principles:
            active_principles = [p for p in self.principles if p["name"] in principles]
        else:
            active_principles = self.principles

        critique_questions = "\n".join(
            f"- {p['name'].title()}: {p['critique']}"
            for p in active_principles
        )

        return f"""Please critique the following response based on these principles:

{critique_questions}

Response to critique:
---
{response}
---

For each principle, provide:
1. Assessment (Good/Needs Improvement/Problematic)
2. Specific issues found (if any)
3. Suggested improvements

Critique:"""

    def create_revision_prompt(
        self,
        original_response: str,
        critique: str,
    ) -> str:
        """
        Create a prompt to revise a response based on critique.

        Args:
            original_response: The original response
            critique: The critique of the response

        Returns:
            Revision prompt
        """
        return f"""Please revise the following response to address the critique below.

Original Response:
---
{original_response}
---

Critique:
---
{critique}
---

Please provide a revised response that addresses the issues identified in the critique while maintaining helpfulness.

Revised Response:"""

    def create_constitutional_chain(
        self,
        query: str,
    ) -> List[Dict[str, str]]:
        """
        Create a constitutional AI prompting chain.

        Returns a list of prompts to:
        1. Generate initial response
        2. Critique the response
        3. Revise based on critique

        Args:
            query: The user's query

        Returns:
            List of prompt dictionaries with 'role' and 'stage'
        """
        return [
            {
                "stage": "generate",
                "prompt": f"Please respond to the following query:\n\n{query}",
            },
            {
                "stage": "critique",
                "prompt": "critique_template",  # Will use create_critique_prompt
            },
            {
                "stage": "revise",
                "prompt": "revision_template",  # Will use create_revision_prompt
            },
        ]


# =============================================================================
# Dynamic Prompt Optimizer
# =============================================================================

class DynamicPromptOptimizer:
    """
    Dynamically optimizes prompts based on task type and context.

    Features:
    - Automatic prompt type selection
    - Context-aware prompt modification
    - Performance-based prompt tuning
    """

    TASK_PATTERNS = {
        "math": [r'\b(calculate|compute|solve|equation|formula)\b', r'\d+\s*[\+\-\*\/]\s*\d+'],
        "code": [r'\b(code|function|implement|program|debug|fix)\b', r'```'],
        "reasoning": [r'\b(why|explain|analyze|compare|evaluate)\b'],
        "creative": [r'\b(write|story|poem|creative|imagine)\b'],
        "factual": [r'\b(what is|who is|when|where|define)\b'],
        "planning": [r'\b(plan|steps|how to|process|procedure)\b'],
    }

    TASK_TO_PROMPT_TYPE = {
        "math": PromptType.CHAIN_OF_THOUGHT,
        "code": PromptType.CHAIN_OF_THOUGHT,
        "reasoning": PromptType.CHAIN_OF_THOUGHT,
        "creative": PromptType.STANDARD,
        "factual": PromptType.STANDARD,
        "planning": PromptType.REACT,
    }

    def __init__(self):
        self.cot = ChainOfThoughtPrompt()
        self.react = ReActPrompt()
        self.constitutional = ConstitutionalPrompt()

        # Performance tracking
        self._task_success: Dict[str, List[bool]] = {}

    def detect_task_type(self, query: str) -> str:
        """Detect the type of task from the query."""
        query_lower = query.lower()

        scores: Dict[str, int] = {}
        for task_type, patterns in self.TASK_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[task_type] = score

        # Return task with highest score, default to "reasoning"
        if max(scores.values()) > 0:
            return max(scores, key=lambda k: scores[k])
        return "reasoning"

    def optimize_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, PromptType]:
        """
        Optimize a prompt for the given query.

        Args:
            query: The user's query
            context: Additional context (conversation history, etc.)
            tools: Available tools for ReAct

        Returns:
            Tuple of (optimized_prompt, prompt_type)
        """
        task_type = self.detect_task_type(query)
        prompt_type = self.TASK_TO_PROMPT_TYPE.get(task_type, PromptType.STANDARD)

        # Override to ReAct if tools are available and task could benefit
        if tools and task_type in ["planning", "factual"]:
            prompt_type = PromptType.REACT

        # Build optimized prompt
        if prompt_type == PromptType.CHAIN_OF_THOUGHT:
            if task_type == "math":
                optimized = self.cot.create_math_cot(query)
            elif task_type == "code":
                optimized = self.cot.create_code_cot(query)
            else:
                optimized = self.cot.wrap_query(query, "structured")

        elif prompt_type == PromptType.REACT:
            if tools:
                self.react.tools = tools
            optimized = self.react.create_react_prompt(query)

        else:
            optimized = query

        return optimized, prompt_type

    def get_system_prompt(
        self,
        prompt_type: PromptType,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Get appropriate system prompt for the prompt type."""
        if prompt_type == PromptType.REACT and tools:
            return self.react.create_system_prompt(tools)

        if prompt_type == PromptType.CHAIN_OF_THOUGHT:
            return """You are a helpful AI assistant that thinks carefully through problems step by step.

When solving problems:
1. Break down complex problems into smaller steps
2. Show your reasoning clearly
3. Verify your answers when possible
4. Acknowledge if you're uncertain

Your goal is to provide accurate, well-reasoned responses."""

        return """You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries."""


# =============================================================================
# Prompt Builder
# =============================================================================

class PromptBuilder:
    """
    High-level prompt builder that combines all prompt engineering techniques.
    """

    def __init__(
        self,
        default_system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        self.default_system_prompt = default_system_prompt
        self.tools = tools or []

        self.optimizer = DynamicPromptOptimizer()
        self.cot = ChainOfThoughtPrompt()
        self.react = ReActPrompt(tools)
        self.constitutional = ConstitutionalPrompt()

        # Template library
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default prompt templates."""
        self._templates["summarize"] = PromptTemplate(
            name="summarize",
            template="Please summarize the following content:\n\n{content}\n\nSummary:",
            variables=["content"],
        )

        self._templates["translate"] = PromptTemplate(
            name="translate",
            template="Please translate the following text from {source_lang} to {target_lang}:\n\n{text}\n\nTranslation:",
            variables=["source_lang", "target_lang", "text"],
        )

        self._templates["explain"] = PromptTemplate(
            name="explain",
            template="Please explain {topic} in simple terms, suitable for {audience}.\n\nExplanation:",
            variables=["topic", "audience"],
        )

    def register_template(self, template: PromptTemplate) -> None:
        """Register a custom template."""
        self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def build(
        self,
        query: str,
        template_name: Optional[str] = None,
        template_vars: Optional[Dict[str, Any]] = None,
        auto_optimize: bool = True,
        use_cot: bool = False,
        use_react: bool = False,
        use_constitutional: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build an optimized prompt.

        Args:
            query: The user's query or input
            template_name: Use a specific template
            template_vars: Variables for the template
            auto_optimize: Automatically select best prompt strategy
            use_cot: Force chain-of-thought prompting
            use_react: Force ReAct prompting
            use_constitutional: Apply constitutional AI
            context: Additional context

        Returns:
            Dict with 'prompt', 'system_prompt', 'prompt_type', 'metadata'
        """
        result = {
            "prompt": query,
            "system_prompt": self.default_system_prompt or "",
            "prompt_type": PromptType.STANDARD,
            "metadata": {},
        }

        # Apply template if specified
        if template_name and template_name in self._templates:
            template = self._templates[template_name]
            result["prompt"] = template.format(**(template_vars or {"content": query}))
            result["prompt_type"] = template.prompt_type

        # Apply specific prompting technique
        if use_react and self.tools:
            result["prompt"] = self.react.create_react_prompt(query)
            result["system_prompt"] = self.react.create_system_prompt()
            result["prompt_type"] = PromptType.REACT

        elif use_cot:
            result["prompt"] = self.cot.wrap_query(query, "structured")
            result["prompt_type"] = PromptType.CHAIN_OF_THOUGHT

        elif auto_optimize:
            optimized, prompt_type = self.optimizer.optimize_prompt(
                query,
                context=context,
                tools=self.tools if self.tools else None,
            )
            result["prompt"] = optimized
            result["prompt_type"] = prompt_type
            result["system_prompt"] = self.optimizer.get_system_prompt(
                prompt_type,
                self.tools if prompt_type == PromptType.REACT else None,
            )

        # Add constitutional wrapper if requested
        if use_constitutional:
            result["metadata"]["constitutional_chain"] = self.constitutional.create_constitutional_chain(
                result["prompt"]
            )

        result["metadata"]["task_type"] = self.optimizer.detect_task_type(query)
        result["metadata"]["tools_available"] = len(self.tools) > 0

        return result
