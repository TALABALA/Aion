"""
AION Goal Executor

Executes goals by:
- Creating execution plans
- Spawning agents
- Orchestrating tools
- Tracking progress
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Optional

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalEvent,
    GoalProgress,
    ExecutionContext,
)

logger = structlog.get_logger(__name__)


class GoalExecutor:
    """
    Executes goals using AION's planning and agent systems.

    Execution flow:
    1. Generate execution plan for goal
    2. Spawn dedicated agent (if needed)
    3. Execute plan steps
    4. Track progress and handle failures
    5. Return results
    """

    def __init__(
        self,
        planning_engine: Optional[Any] = None,
        process_supervisor: Optional[Any] = None,
        tool_orchestrator: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
        max_retries: int = 3,
        step_timeout: float = 300.0,  # 5 minutes per step
    ):
        self._planning = planning_engine
        self._supervisor = process_supervisor
        self._tools = tool_orchestrator
        self._llm = llm_provider

        self._max_retries = max_retries
        self._step_timeout = step_timeout

        # Active executions
        self._active_executions: dict[str, ExecutionContext] = {}

        # Callbacks
        self._progress_callbacks: list[Callable[[GoalProgress], None]] = []

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the executor."""
        if self._initialized:
            return

        logger.info("Initializing Goal Executor")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the executor."""
        logger.info("Shutting down Goal Executor")

        # Cancel all active executions
        for ctx in self._active_executions.values():
            ctx.cancelled = True

        self._initialized = False

    async def execute(self, goal: Goal) -> dict[str, Any]:
        """
        Execute a goal.

        Args:
            goal: The goal to execute

        Returns:
            Execution result with success status, outcome, artifacts
        """
        logger.info(f"Executing goal: {goal.title}", goal_id=goal.id[:8])

        start_time = datetime.now()
        result = {
            "success": False,
            "outcome": "",
            "artifacts": [],
            "metrics": {},
            "learnings": [],
        }

        # Create execution context
        ctx = ExecutionContext(
            goal=goal,
            max_retries=self._max_retries,
        )

        # Set resource limits from goal constraints
        for constraint in goal.constraints:
            if constraint.max_tokens:
                ctx.tokens_budget = constraint.max_tokens
            if constraint.max_cost_dollars:
                ctx.cost_budget = constraint.max_cost_dollars
            if constraint.max_duration_hours:
                ctx.timeout_seconds = constraint.max_duration_hours * 3600

        self._active_executions[goal.id] = ctx

        try:
            # Phase 1: Generate execution plan
            plan = await self._generate_plan(goal, ctx)
            if not plan:
                result["outcome"] = "Failed to generate execution plan"
                result["reason"] = "Planning failed"
                return result

            goal.plan_id = plan.get("id")

            # Phase 2: Execute plan
            execution_result = await self._execute_plan(goal, plan, ctx)

            # Phase 3: Verify success criteria
            criteria_met = await self._verify_success_criteria(goal, execution_result)

            result["success"] = criteria_met
            result["outcome"] = execution_result.get("outcome", "")
            result["artifacts"] = execution_result.get("artifacts", [])
            result["learnings"] = execution_result.get("learnings", [])
            result["metrics"] = {
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "steps_executed": execution_result.get("steps_completed", 0),
                "tools_used": execution_result.get("tools_used", []),
                "tokens_used": ctx.tokens_used,
                "cost_used": ctx.cost_used,
                "retries": ctx.retry_count,
            }

            return result

        except asyncio.CancelledError:
            logger.info(f"Goal execution cancelled", goal_id=goal.id[:8])
            result["outcome"] = "Execution cancelled"
            result["reason"] = "Cancelled"
            return result

        except Exception as e:
            logger.error(f"Goal execution failed: {e}", goal_id=goal.id[:8])
            result["outcome"] = str(e)
            result["reason"] = "Execution error"
            return result

        finally:
            self._active_executions.pop(goal.id, None)

    async def _generate_plan(
        self, goal: Goal, ctx: ExecutionContext
    ) -> Optional[dict[str, Any]]:
        """Generate an execution plan for the goal."""
        if self._planning:
            try:
                # Use planning engine to create plan
                plan = await self._planning.create_plan(
                    objective=goal.title,
                    description=goal.description,
                    success_criteria=goal.success_criteria,
                    constraints=[c.description for c in goal.constraints],
                )
                return {"id": plan.id, "nodes": plan.nodes, "plan": plan}

            except Exception as e:
                logger.error(f"Plan generation failed: {e}")

        # Create a simple plan without planning engine
        return self._create_simple_plan(goal)

    def _create_simple_plan(self, goal: Goal) -> dict[str, Any]:
        """Create a simple single-step plan."""
        import uuid

        return {
            "id": str(uuid.uuid4()),
            "name": f"Plan for: {goal.title}",
            "description": goal.description,
            "nodes": [
                {
                    "id": "execute",
                    "name": "Execute Goal",
                    "description": f"Execute: {goal.description}",
                    "action_type": "llm_reason",
                    "parameters": {
                        "goal": goal.title,
                        "criteria": goal.success_criteria,
                    },
                }
            ],
        }

    async def _execute_plan(
        self,
        goal: Goal,
        plan: dict[str, Any],
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute the plan for a goal."""
        result = {
            "outcome": "",
            "artifacts": [],
            "steps_completed": 0,
            "tools_used": [],
            "learnings": [],
        }

        nodes = plan.get("nodes", [])

        for i, node in enumerate(nodes):
            if ctx.cancelled:
                result["outcome"] = "Execution cancelled"
                break

            if ctx.paused:
                result["outcome"] = "Execution paused"
                break

            if not ctx.is_within_budget():
                result["outcome"] = "Resource budget exceeded"
                break

            if ctx.is_timed_out():
                result["outcome"] = "Execution timed out"
                break

            # Execute node
            try:
                node_result = await self._execute_node(goal, node, ctx)

                if node_result.get("success"):
                    result["steps_completed"] += 1

                    if node_result.get("artifacts"):
                        result["artifacts"].extend(node_result["artifacts"])

                    if node_result.get("tools_used"):
                        result["tools_used"].extend(node_result["tools_used"])

                    # Update progress
                    progress = (i + 1) / len(nodes) * 100
                    await self._report_progress(
                        goal,
                        progress,
                        f"Completed step: {node.get('name', i+1)}",
                    )
                else:
                    # Handle failure
                    if ctx.retry_count < ctx.max_retries:
                        ctx.retry_count += 1
                        logger.warning(
                            f"Step failed, retrying ({ctx.retry_count}/{ctx.max_retries})",
                            goal_id=goal.id[:8],
                        )
                        # Retry the same node
                        continue
                    else:
                        result["outcome"] = f"Step failed: {node_result.get('error', 'Unknown error')}"
                        break

            except asyncio.TimeoutError:
                result["outcome"] = f"Step timed out: {node.get('name', i+1)}"
                break

            except Exception as e:
                logger.error(f"Step execution error: {e}")
                result["outcome"] = f"Step error: {e}"
                break

        if not result["outcome"]:
            result["outcome"] = "Plan executed successfully"

        return result

    async def _execute_node(
        self,
        goal: Goal,
        node: dict[str, Any],
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute a single plan node."""
        action_type = node.get("action_type", "llm_reason")
        parameters = node.get("parameters", {})

        result = {
            "success": False,
            "output": "",
            "artifacts": [],
            "tools_used": [],
        }

        try:
            async with asyncio.timeout(self._step_timeout):
                if action_type == "llm_reason":
                    result = await self._execute_reasoning(goal, node, ctx)

                elif action_type == "tool_use":
                    result = await self._execute_tool(node, ctx)

                elif action_type == "agent_spawn":
                    result = await self._execute_agent_spawn(goal, node, ctx)

                else:
                    # Default to reasoning
                    result = await self._execute_reasoning(goal, node, ctx)

        except asyncio.TimeoutError:
            result["error"] = "Execution timed out"

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Node execution error: {e}")

        return result

    async def _execute_reasoning(
        self,
        goal: Goal,
        node: dict[str, Any],
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute a reasoning step using LLM."""
        if not self._llm:
            return {
                "success": True,
                "output": "Reasoning step completed (no LLM available)",
                "artifacts": [],
                "tools_used": [],
            }

        system_prompt = f"""You are executing a step toward achieving a goal.

Goal: {goal.title}
Description: {goal.description}
Success Criteria: {goal.success_criteria}

Current Step: {node.get('name', 'Execute')}
Step Description: {node.get('description', '')}
"""

        user_prompt = f"""
Execute this step and provide the result.

Parameters: {node.get('parameters', {})}

Provide your response in JSON format:
{{
    "action_taken": "...",
    "output": "...",
    "artifacts_created": [],
    "next_steps": [],
    "success": true/false
}}
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            # Track token usage
            if hasattr(response, "input_tokens"):
                ctx.tokens_used += response.input_tokens
            if hasattr(response, "output_tokens"):
                ctx.tokens_used += response.output_tokens

            # Parse response
            response_text = ""
            if hasattr(response, "content"):
                for block in response.content:
                    if hasattr(block, "text"):
                        response_text = block.text
                        break

            # Try to parse as JSON
            try:
                import json

                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(response_text[json_start:json_end])
                    return {
                        "success": data.get("success", True),
                        "output": data.get("output", response_text),
                        "artifacts": data.get("artifacts_created", []),
                        "tools_used": [],
                    }
            except:
                pass

            return {
                "success": True,
                "output": response_text,
                "artifacts": [],
                "tools_used": [],
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "artifacts": [],
                "tools_used": [],
            }

    async def _execute_tool(
        self,
        node: dict[str, Any],
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute a tool use step."""
        if not self._tools:
            return {
                "success": False,
                "error": "Tool orchestrator not available",
                "artifacts": [],
                "tools_used": [],
            }

        tool_name = node.get("parameters", {}).get("tool_name", "")
        tool_params = node.get("parameters", {}).get("tool_parameters", {})

        try:
            result = await self._tools.execute_tool(tool_name, tool_params)

            goal = ctx.goal
            goal.metrics.tool_calls += 1

            return {
                "success": result.get("success", True),
                "output": result.get("output", ""),
                "artifacts": result.get("artifacts", []),
                "tools_used": [tool_name],
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "artifacts": [],
                "tools_used": [tool_name],
            }

    async def _execute_agent_spawn(
        self,
        goal: Goal,
        node: dict[str, Any],
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Spawn an agent for complex execution."""
        if not self._supervisor:
            return {
                "success": False,
                "error": "Process supervisor not available",
                "artifacts": [],
                "tools_used": [],
            }

        try:
            agent_id = await self.spawn_goal_agent(goal)
            ctx.agent_id = agent_id
            goal.agent_id = agent_id

            return {
                "success": True,
                "output": f"Agent spawned: {agent_id}",
                "artifacts": [],
                "tools_used": ["agent_spawn"],
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "artifacts": [],
                "tools_used": [],
            }

    async def _verify_success_criteria(
        self,
        goal: Goal,
        execution_result: dict[str, Any],
    ) -> bool:
        """Verify that success criteria are met."""
        if not goal.success_criteria:
            # No specific criteria, assume success if execution completed
            return "error" not in execution_result.get("outcome", "").lower()

        if not self._llm:
            # Can't verify without LLM
            return "error" not in execution_result.get("outcome", "").lower()

        # Use LLM to verify criteria
        system_prompt = """You are verifying if a goal's success criteria have been met.
Be strict but fair in your assessment."""

        criteria_list = "\n".join(f"- {c}" for c in goal.success_criteria)
        user_prompt = f"""
Verify if these success criteria have been met:

{criteria_list}

Execution result:
{execution_result.get('outcome', '')}

Artifacts created:
{execution_result.get('artifacts', [])}

Respond in JSON format:
{{
    "all_criteria_met": true/false,
    "criteria_results": [
        {{"criterion": "...", "met": true/false, "reasoning": "..."}}
    ],
    "overall_assessment": "..."
}}
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            response_text = ""
            if hasattr(response, "content"):
                for block in response.content:
                    if hasattr(block, "text"):
                        response_text = block.text
                        break

            import json

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return data.get("all_criteria_met", False)

        except Exception as e:
            logger.error(f"Failed to verify criteria: {e}")

        return "error" not in execution_result.get("outcome", "").lower()

    async def spawn_goal_agent(self, goal: Goal) -> Optional[str]:
        """Spawn a dedicated agent for goal execution."""
        if not self._supervisor:
            return None

        try:
            from aion.systems.process.models import (
                AgentConfig,
                ProcessPriority,
                RestartPolicy,
            )
            from aion.systems.goals.types import GoalPriority

            priority_mapping = {
                GoalPriority.CRITICAL: ProcessPriority.CRITICAL,
                GoalPriority.HIGH: ProcessPriority.HIGH,
                GoalPriority.MEDIUM: ProcessPriority.NORMAL,
                GoalPriority.LOW: ProcessPriority.LOW,
                GoalPriority.BACKGROUND: ProcessPriority.IDLE,
            }

            config = AgentConfig(
                name=f"goal_agent_{goal.id[:8]}",
                agent_class="goal_executor",
                system_prompt=self._build_agent_prompt(goal),
                tools=["*"],  # All tools
                memory_enabled=True,
                planning_enabled=True,
                initial_goal=goal.title,
                instructions=goal.success_criteria,
                priority=priority_mapping.get(goal.priority, ProcessPriority.NORMAL),
                restart_policy=RestartPolicy.ON_FAILURE,
                metadata={
                    "goal_id": goal.id,
                    "goal_title": goal.title,
                },
            )

            agent_id = await self._supervisor.spawn_agent(config)
            return agent_id

        except ImportError:
            logger.warning("Process models not available")
            return None

        except Exception as e:
            logger.error(f"Failed to spawn goal agent: {e}")
            return None

    def _build_agent_prompt(self, goal: Goal) -> str:
        """Build system prompt for goal agent."""
        criteria = "\n".join(f"- {c}" for c in goal.success_criteria)
        return f"""You are an autonomous agent working toward a specific goal.

## Your Goal
{goal.title}

## Description
{goal.description}

## Success Criteria
{criteria}

## Instructions
1. Break down the goal into actionable steps
2. Use available tools to make progress
3. Track your progress toward each success criterion
4. Report when you've completed the goal or if you're blocked

Work autonomously but stay focused on the goal. If you encounter obstacles, try alternative approaches before reporting failure."""

    async def _report_progress(
        self,
        goal: Goal,
        progress_percent: float,
        message: str,
    ) -> None:
        """Report progress on a goal."""
        progress = GoalProgress(
            goal_id=goal.id,
            progress_percent=progress_percent,
            status=GoalStatus.ACTIVE,
            message=message,
        )

        # Update goal metrics
        goal.metrics.update_progress(progress_percent)

        # Notify callbacks
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def pause_execution(self, goal_id: str) -> bool:
        """Pause execution of a goal."""
        ctx = self._active_executions.get(goal_id)
        if ctx:
            ctx.paused = True
            return True
        return False

    def resume_execution(self, goal_id: str) -> bool:
        """Resume execution of a goal."""
        ctx = self._active_executions.get(goal_id)
        if ctx:
            ctx.paused = False
            return True
        return False

    def cancel_execution(self, goal_id: str) -> bool:
        """Cancel execution of a goal."""
        ctx = self._active_executions.get(goal_id)
        if ctx:
            ctx.cancelled = True
            return True
        return False

    def get_active_executions(self) -> list[str]:
        """Get IDs of active executions."""
        return list(self._active_executions.keys())

    def on_progress(self, callback: Callable[[GoalProgress], None]) -> None:
        """Register progress callback."""
        self._progress_callbacks.append(callback)
