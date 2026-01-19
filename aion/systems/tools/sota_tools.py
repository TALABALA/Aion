"""
AION SOTA Tool System

Toolformer-inspired tool learning with:
- Self-taught tool use detection
- Tool output verification
- Self-correction on failures
- Dynamic tool composition
- Tool learning from demonstrations
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Tool Use Detection (Toolformer-style)
# ============================================================================

@dataclass
class ToolCall:
    """A parsed tool call."""
    tool_name: str
    arguments: dict[str, Any]
    raw_text: str
    position: int  # Position in text


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    verified: bool = False
    verification_score: float = 0.0


class ToolUseDetector:
    """
    Detects when tool use would be beneficial in text generation.

    Inspired by Toolformer: learns to insert tool calls at
    positions where they would improve the output.
    """

    def __init__(self, llm_adapter, available_tools: list[dict]):
        self.llm = llm_adapter
        self.available_tools = available_tools

        # Learning from successful tool uses
        self.tool_use_examples: list[dict] = []
        self.tool_success_rates: dict[str, list[bool]] = defaultdict(list)

    async def should_use_tool(
        self,
        text_so_far: str,
        query: str,
    ) -> Optional[ToolCall]:
        """
        Determine if a tool should be used at this point.

        Args:
            text_so_far: Generated text so far
            query: Original query

        Returns:
            ToolCall if tool use is beneficial, None otherwise
        """
        from aion.core.llm import Message

        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in self.available_tools
        ])

        prompt = f"""You are deciding whether to use a tool to help answer a question.

Question: {query}

Text generated so far: {text_so_far}

Available tools:
{tools_desc}

Should a tool be used here? Consider:
1. Would a tool provide information not already available?
2. Is the tool appropriate for what's needed?
3. Would the tool result improve the answer quality?

If YES, respond with:
USE_TOOL: <tool_name>
ARGUMENTS: <json arguments>
REASON: <why this tool helps>

If NO, respond with:
NO_TOOL_NEEDED
REASON: <why no tool is needed>
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You decide when to use tools strategically."),
                Message(role="user", content=prompt),
            ], temperature=0.2)

            content = response.content

            if "USE_TOOL:" in content:
                # Parse tool call
                tool_match = re.search(r'USE_TOOL:\s*(\w+)', content)
                args_match = re.search(r'ARGUMENTS:\s*({.*?})', content, re.DOTALL)

                if tool_match:
                    tool_name = tool_match.group(1)
                    try:
                        arguments = json.loads(args_match.group(1)) if args_match else {}
                    except:
                        arguments = {}

                    return ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        raw_text=content,
                        position=len(text_so_far),
                    )

            return None

        except Exception as e:
            logger.warning("Tool detection failed", error=str(e))
            return None

    def record_tool_use(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        was_helpful: bool,
    ) -> None:
        """Record a tool use for learning."""
        self.tool_use_examples.append({
            "tool": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "success": result.success,
            "helpful": was_helpful,
            "timestamp": datetime.now(),
        })

        self.tool_success_rates[tool_call.tool_name].append(was_helpful)

        # Keep only recent examples
        if len(self.tool_use_examples) > 1000:
            self.tool_use_examples = self.tool_use_examples[-1000:]

    def get_tool_stats(self) -> dict[str, float]:
        """Get success rates for each tool."""
        stats = {}
        for tool, successes in self.tool_success_rates.items():
            if successes:
                stats[tool] = sum(successes) / len(successes)
        return stats


# ============================================================================
# Tool Output Verification
# ============================================================================

class ToolVerifier:
    """
    Verifies tool outputs for correctness.

    Uses multiple verification strategies:
    - Self-consistency checking
    - Cross-validation with other tools
    - LLM-based fact checking
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    async def verify(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        context: str,
    ) -> tuple[bool, float, str]:
        """
        Verify a tool result.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            context: Context in which tool was called

        Returns:
            Tuple of (is_valid, confidence, explanation)
        """
        from aion.core.llm import Message

        prompt = f"""Verify the following tool result for accuracy and relevance.

Context: {context}

Tool: {tool_name}
Arguments: {json.dumps(arguments)}
Result: {str(result)[:1000]}

Verification checklist:
1. Is the result format correct for this tool?
2. Does the result make logical sense?
3. Is the result relevant to the context?
4. Are there any obvious errors or inconsistencies?

Respond with:
VALID: YES or NO
CONFIDENCE: 0-100
ISSUES: List any problems (or "None")
EXPLANATION: Brief explanation
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You verify tool outputs for accuracy."),
                Message(role="user", content=prompt),
            ], temperature=0.1)

            content = response.content

            is_valid = "VALID: YES" in content.upper()

            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', content)
            confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5

            explanation = content

            return is_valid, confidence, explanation

        except Exception as e:
            logger.warning("Tool verification failed", error=str(e))
            return True, 0.5, "Verification failed, assuming valid"

    async def cross_validate(
        self,
        tool_results: list[ToolResult],
    ) -> dict[str, bool]:
        """
        Cross-validate multiple tool results for consistency.

        Returns:
            Dict mapping result ID to validity
        """
        if len(tool_results) < 2:
            return {str(i): True for i in range(len(tool_results))}

        from aion.core.llm import Message

        results_text = "\n".join([
            f"Result {i+1} ({r.tool_name}): {str(r.result)[:200]}"
            for i, r in enumerate(tool_results)
        ])

        prompt = f"""Check these tool results for consistency with each other.

Results:
{results_text}

For each result, indicate if it's consistent with the others:
Result 1: CONSISTENT or INCONSISTENT
Result 2: CONSISTENT or INCONSISTENT
...

Also explain any inconsistencies found.
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You check results for consistency."),
                Message(role="user", content=prompt),
            ])

            validity = {}
            for i in range(len(tool_results)):
                pattern = f"Result {i+1}:.*?(CONSISTENT|INCONSISTENT)"
                match = re.search(pattern, response.content, re.IGNORECASE)
                validity[str(i)] = match and "INCONSISTENT" not in match.group(1).upper()

            return validity

        except:
            return {str(i): True for i in range(len(tool_results))}


# ============================================================================
# Self-Correction System
# ============================================================================

class ToolSelfCorrector:
    """
    Self-correction for failed tool executions.

    Implements:
    - Error analysis
    - Argument correction
    - Alternative tool selection
    - Retry with modifications
    """

    def __init__(
        self,
        llm_adapter,
        tool_executor: Callable,
        max_retries: int = 3,
    ):
        self.llm = llm_adapter
        self.tool_executor = tool_executor
        self.max_retries = max_retries

        # Learning from corrections
        self.correction_history: list[dict] = []

    async def execute_with_correction(
        self,
        tool_name: str,
        arguments: dict,
        context: str,
        available_tools: list[dict],
    ) -> ToolResult:
        """
        Execute a tool with automatic self-correction on failure.

        Args:
            tool_name: Tool to execute
            arguments: Tool arguments
            context: Execution context
            available_tools: All available tools

        Returns:
            ToolResult after potential corrections
        """
        attempts = []

        for attempt in range(self.max_retries):
            # Execute tool
            try:
                result = await self.tool_executor(tool_name, arguments)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)

            attempts.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "success": success,
                "error": error,
            })

            if success:
                return ToolResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    success=True,
                )

            # Analyze failure and correct
            correction = await self._analyze_and_correct(
                attempts[-1],
                context,
                available_tools,
            )

            if correction is None:
                break

            # Apply correction
            tool_name = correction.get("tool", tool_name)
            arguments = correction.get("arguments", arguments)

            logger.info(
                "Self-correction applied",
                attempt=attempt + 1,
                new_tool=tool_name,
            )

        # Record for learning
        self.correction_history.append({
            "attempts": attempts,
            "final_success": attempts[-1]["success"] if attempts else False,
        })

        # Return last attempt
        last = attempts[-1] if attempts else {"result": None, "error": "No attempts"}
        return ToolResult(
            tool_name=tool_name,
            arguments=arguments,
            result=last.get("result"),
            success=last.get("success", False),
            error=last.get("error"),
        )

    async def _analyze_and_correct(
        self,
        failed_attempt: dict,
        context: str,
        available_tools: list[dict],
    ) -> Optional[dict]:
        """Analyze failure and suggest correction."""
        from aion.core.llm import Message

        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in available_tools
        ])

        prompt = f"""A tool execution failed. Analyze and suggest a correction.

Context: {context}

Failed attempt:
- Tool: {failed_attempt['tool']}
- Arguments: {json.dumps(failed_attempt['arguments'])}
- Error: {failed_attempt['error']}

Available tools:
{tools_desc}

Analyze the failure and suggest a fix:
1. What went wrong?
2. Should we use a different tool?
3. Should we modify the arguments?
4. Is this task achievable with available tools?

Respond with:
ANALYSIS: <what went wrong>
ACTION: RETRY_SAME, TRY_DIFFERENT_TOOL, MODIFY_ARGUMENTS, or GIVE_UP
NEW_TOOL: <tool name if different>
NEW_ARGUMENTS: <json arguments if modified>
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You analyze tool failures and suggest corrections."),
                Message(role="user", content=prompt),
            ], temperature=0.2)

            content = response.content

            if "GIVE_UP" in content.upper():
                return None

            correction = {}

            if "TRY_DIFFERENT_TOOL" in content.upper():
                tool_match = re.search(r'NEW_TOOL:\s*(\w+)', content)
                if tool_match:
                    correction["tool"] = tool_match.group(1)

            args_match = re.search(r'NEW_ARGUMENTS:\s*({.*?})', content, re.DOTALL)
            if args_match:
                try:
                    correction["arguments"] = json.loads(args_match.group(1))
                except:
                    pass

            return correction if correction else None

        except Exception as e:
            logger.warning("Correction analysis failed", error=str(e))
            return None


# ============================================================================
# Dynamic Tool Composition
# ============================================================================

@dataclass
class ToolPipeline:
    """A composed pipeline of tools."""
    id: str
    name: str
    steps: list[dict]  # [{tool, arg_template, output_name}]
    success_count: int = 0
    failure_count: int = 0


class ToolComposer:
    """
    Dynamically composes tools into pipelines.

    Learns effective tool combinations from experience.
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Learned pipelines
        self.pipelines: dict[str, ToolPipeline] = {}
        self.task_to_pipeline: dict[str, str] = {}  # task_type -> pipeline_id

        # Co-occurrence statistics
        self.tool_cooccurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    async def compose_pipeline(
        self,
        task: str,
        available_tools: list[dict],
        max_steps: int = 5,
    ) -> ToolPipeline:
        """
        Compose a tool pipeline for a task.

        Args:
            task: Task description
            available_tools: Available tools
            max_steps: Maximum pipeline steps

        Returns:
            Composed ToolPipeline
        """
        from aion.core.llm import Message

        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}\n  Parameters: {t.get('parameters', {})}"
            for t in available_tools
        ])

        prompt = f"""Design a tool pipeline to accomplish this task.

Task: {task}

Available tools:
{tools_desc}

Design a sequence of tool calls (max {max_steps} steps) that accomplishes the task.
Each step can use the output of previous steps.

Format:
STEP 1:
TOOL: <tool_name>
ARGUMENTS: <json with $prev for previous output>
OUTPUT_NAME: <name for this output>
---
STEP 2:
...

Example argument using previous output: {{"query": "$step1_output"}}
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You design efficient tool pipelines."),
                Message(role="user", content=prompt),
            ], temperature=0.3)

            steps = self._parse_pipeline(response.content)

            pipeline = ToolPipeline(
                id=str(uuid.uuid4()),
                name=f"Pipeline for: {task[:50]}",
                steps=steps,
            )

            self.pipelines[pipeline.id] = pipeline

            return pipeline

        except Exception as e:
            logger.warning("Pipeline composition failed", error=str(e))
            return ToolPipeline(id=str(uuid.uuid4()), name="Empty", steps=[])

    def _parse_pipeline(self, content: str) -> list[dict]:
        """Parse pipeline from LLM response."""
        steps = []
        current_step = {}

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("TOOL:"):
                if current_step:
                    steps.append(current_step)
                current_step = {"tool": line.replace("TOOL:", "").strip()}

            elif line.startswith("ARGUMENTS:"):
                args_str = line.replace("ARGUMENTS:", "").strip()
                try:
                    current_step["arguments"] = json.loads(args_str)
                except:
                    current_step["arguments"] = {}

            elif line.startswith("OUTPUT_NAME:"):
                current_step["output_name"] = line.replace("OUTPUT_NAME:", "").strip()

        if current_step:
            steps.append(current_step)

        return steps

    async def execute_pipeline(
        self,
        pipeline: ToolPipeline,
        tool_executor: Callable,
        initial_context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Execute a tool pipeline.

        Args:
            pipeline: Pipeline to execute
            tool_executor: Tool execution function
            initial_context: Initial context variables

        Returns:
            Dict of all outputs
        """
        context = initial_context or {}
        outputs = {}

        for i, step in enumerate(pipeline.steps):
            tool_name = step.get("tool")
            arg_template = step.get("arguments", {})
            output_name = step.get("output_name", f"step{i+1}_output")

            # Substitute context variables
            arguments = self._substitute_variables(arg_template, context)

            try:
                result = await tool_executor(tool_name, arguments)
                outputs[output_name] = result
                context[output_name] = result

                # Record co-occurrence
                if i > 0:
                    prev_tool = pipeline.steps[i-1].get("tool")
                    self.tool_cooccurrence[prev_tool][tool_name] += 1

            except Exception as e:
                logger.warning(f"Pipeline step {i+1} failed", error=str(e))
                outputs[output_name] = {"error": str(e)}
                pipeline.failure_count += 1
                return outputs

        pipeline.success_count += 1
        return outputs

    def _substitute_variables(
        self,
        template: dict,
        context: dict,
    ) -> dict:
        """Substitute $variables in argument template."""
        result = {}

        for key, value in template.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                result[key] = context.get(var_name, value)
            elif isinstance(value, dict):
                result[key] = self._substitute_variables(value, context)
            else:
                result[key] = value

        return result

    def suggest_next_tool(self, current_tool: str) -> list[tuple[str, float]]:
        """Suggest next tool based on co-occurrence statistics."""
        if current_tool not in self.tool_cooccurrence:
            return []

        cooccur = self.tool_cooccurrence[current_tool]
        total = sum(cooccur.values())

        if total == 0:
            return []

        suggestions = [
            (tool, count / total)
            for tool, count in cooccur.items()
        ]

        return sorted(suggestions, key=lambda x: x[1], reverse=True)[:5]


# ============================================================================
# Tool Learning from Demonstrations
# ============================================================================

class ToolDemonstrationLearner:
    """
    Learns tool use from demonstrations.

    Extracts patterns from successful tool use examples
    to improve future tool selection and argument generation.
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Demonstration storage
        self.demonstrations: list[dict] = []
        self.task_patterns: dict[str, list[dict]] = defaultdict(list)

    def add_demonstration(
        self,
        task: str,
        tool_calls: list[ToolCall],
        results: list[ToolResult],
        was_successful: bool,
    ) -> None:
        """Add a demonstration of tool use."""
        demo = {
            "task": task,
            "tool_calls": [
                {"tool": tc.tool_name, "arguments": tc.arguments}
                for tc in tool_calls
            ],
            "results": [
                {"success": r.success, "result": str(r.result)[:200]}
                for r in results
            ],
            "successful": was_successful,
            "timestamp": datetime.now(),
        }

        self.demonstrations.append(demo)

        # Extract task pattern
        task_type = self._extract_task_type(task)
        if was_successful:
            self.task_patterns[task_type].append({
                "tools": [tc.tool_name for tc in tool_calls],
                "example_task": task,
            })

    def _extract_task_type(self, task: str) -> str:
        """Extract general task type from task description."""
        task_lower = task.lower()

        if any(w in task_lower for w in ["search", "find", "look up"]):
            return "search"
        elif any(w in task_lower for w in ["calculate", "compute", "math"]):
            return "calculation"
        elif any(w in task_lower for w in ["fetch", "get", "retrieve", "download"]):
            return "retrieval"
        elif any(w in task_lower for w in ["analyze", "understand", "explain"]):
            return "analysis"
        else:
            return "general"

    async def suggest_from_demonstrations(
        self,
        task: str,
        available_tools: list[dict],
    ) -> list[dict]:
        """
        Suggest tool use based on similar demonstrations.

        Args:
            task: Current task
            available_tools: Available tools

        Returns:
            Suggested tool calls
        """
        task_type = self._extract_task_type(task)
        patterns = self.task_patterns.get(task_type, [])

        if not patterns:
            return []

        from aion.core.llm import Message

        examples = "\n\n".join([
            f"Task: {p['example_task']}\nTools used: {', '.join(p['tools'])}"
            for p in patterns[:5]
        ])

        prompt = f"""Based on these successful examples, suggest tool use for a new task.

Previous successful examples:
{examples}

New task: {task}

Available tools: {', '.join(t['name'] for t in available_tools)}

Suggest which tools to use and in what order.
Format as JSON array: [{{"tool": "name", "arguments": {{}}}}]
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You suggest tool use based on examples."),
                Message(role="user", content=prompt),
            ])

            # Parse JSON
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except:
            pass

        return []


# ============================================================================
# SOTA Tool Orchestrator
# ============================================================================

class SOTAToolOrchestrator:
    """
    State-of-the-art tool orchestration combining:
    - Toolformer-style tool use detection
    - Output verification
    - Self-correction
    - Dynamic composition
    - Learning from demonstrations
    """

    def __init__(
        self,
        llm_adapter,
        tools: list[dict],
        tool_executor: Callable,
    ):
        self.llm = llm_adapter
        self.tools = tools
        self.tool_executor = tool_executor

        # Components
        self.detector = ToolUseDetector(llm_adapter, tools)
        self.verifier = ToolVerifier(llm_adapter)
        self.corrector = ToolSelfCorrector(llm_adapter, tool_executor)
        self.composer = ToolComposer(llm_adapter)
        self.learner = ToolDemonstrationLearner(llm_adapter)

        # Statistics
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "corrections_applied": 0,
            "pipelines_created": 0,
        }

    async def execute(
        self,
        tool_name: str,
        arguments: dict,
        context: str = "",
        verify: bool = True,
        allow_correction: bool = True,
    ) -> ToolResult:
        """
        Execute a tool with verification and correction.

        Args:
            tool_name: Tool to execute
            arguments: Tool arguments
            context: Execution context
            verify: Whether to verify output
            allow_correction: Whether to allow self-correction

        Returns:
            ToolResult
        """
        self._stats["total_executions"] += 1

        if allow_correction:
            result = await self.corrector.execute_with_correction(
                tool_name, arguments, context, self.tools
            )
            if not result.success:
                self._stats["corrections_applied"] += 1
        else:
            try:
                output = await self.tool_executor(tool_name, arguments)
                result = ToolResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=output,
                    success=True,
                )
            except Exception as e:
                result = ToolResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    success=False,
                    error=str(e),
                )

        # Verify if successful
        if verify and result.success:
            is_valid, confidence, explanation = await self.verifier.verify(
                tool_name, arguments, result.result, context
            )
            result.verified = is_valid
            result.verification_score = confidence

        if result.success:
            self._stats["successful_executions"] += 1

        return result

    async def smart_execute(
        self,
        query: str,
        text_so_far: str = "",
    ) -> Optional[ToolResult]:
        """
        Intelligently decide whether to use a tool and execute.

        Args:
            query: User query
            text_so_far: Text generated so far

        Returns:
            ToolResult if tool was used, None otherwise
        """
        # Detect if tool use is beneficial
        tool_call = await self.detector.should_use_tool(text_so_far, query)

        if tool_call is None:
            return None

        # Execute with verification
        result = await self.execute(
            tool_call.tool_name,
            tool_call.arguments,
            context=query,
        )

        # Record for learning
        self.detector.record_tool_use(
            tool_call, result, result.success and result.verified
        )

        return result

    async def execute_composite(
        self,
        task: str,
        max_steps: int = 5,
    ) -> dict[str, Any]:
        """
        Execute a composite task using tool pipeline.

        Args:
            task: Task description
            max_steps: Maximum pipeline steps

        Returns:
            Pipeline outputs
        """
        # First check demonstrations
        suggestions = await self.learner.suggest_from_demonstrations(task, self.tools)

        if suggestions:
            # Execute suggested tools
            results = {}
            tool_calls = []
            tool_results = []

            for sugg in suggestions:
                tool_call = ToolCall(
                    tool_name=sugg["tool"],
                    arguments=sugg.get("arguments", {}),
                    raw_text="",
                    position=0,
                )
                tool_calls.append(tool_call)

                result = await self.execute(sugg["tool"], sugg.get("arguments", {}))
                tool_results.append(result)
                results[sugg["tool"]] = result.result

            # Record demonstration
            self.learner.add_demonstration(
                task, tool_calls, tool_results,
                all(r.success for r in tool_results)
            )

            return results

        # Compose new pipeline
        pipeline = await self.composer.compose_pipeline(task, self.tools, max_steps)
        self._stats["pipelines_created"] += 1

        # Execute pipeline
        outputs = await self.composer.execute_pipeline(
            pipeline, self.tool_executor
        )

        return outputs

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            "tool_success_rates": self.detector.get_tool_stats(),
            "learned_pipelines": len(self.composer.pipelines),
            "demonstrations": len(self.learner.demonstrations),
        }
