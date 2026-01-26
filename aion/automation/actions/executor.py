"""
AION Action Executor

Executes workflow actions of all types.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.automation.types import ActionConfig, ActionType

if TYPE_CHECKING:
    from aion.automation.execution.context import ExecutionContext

logger = structlog.get_logger(__name__)


class ActionExecutor:
    """
    Executes workflow actions.

    Supports:
    - Tool execution
    - Agent operations
    - Goal management
    - Webhooks
    - Notifications
    - Data operations
    - Sub-workflows
    - LLM completions
    - Delays
    - Transforms
    - Scripts
    """

    def __init__(self):
        self._handlers: Dict[ActionType, "BaseActionHandler"] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the action executor."""
        if self._initialized:
            return

        # Initialize action handlers
        from aion.automation.actions.tool import ToolActionHandler
        from aion.automation.actions.agent import AgentActionHandler
        from aion.automation.actions.goal import GoalActionHandler
        from aion.automation.actions.webhook import WebhookActionHandler
        from aion.automation.actions.notification import NotificationActionHandler
        from aion.automation.actions.data import DataActionHandler
        from aion.automation.actions.workflow import WorkflowActionHandler

        self._handlers[ActionType.TOOL] = ToolActionHandler()
        self._handlers[ActionType.AGENT] = AgentActionHandler()
        self._handlers[ActionType.GOAL] = GoalActionHandler()
        self._handlers[ActionType.WEBHOOK] = WebhookActionHandler()
        self._handlers[ActionType.NOTIFICATION] = NotificationActionHandler()
        self._handlers[ActionType.DATA] = DataActionHandler()
        self._handlers[ActionType.WORKFLOW] = WorkflowActionHandler()

        # Built-in handlers
        self._handlers[ActionType.DELAY] = DelayActionHandler()
        self._handlers[ActionType.TRANSFORM] = TransformActionHandler()
        self._handlers[ActionType.LLM] = LLMActionHandler()
        self._handlers[ActionType.SCRIPT] = ScriptActionHandler()

        self._initialized = True
        logger.info("Action executor initialized")

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """
        Execute an action.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Action result
        """
        handler = self._handlers.get(action.action_type)
        if not handler:
            raise ValueError(f"Unknown action type: {action.action_type}")

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                handler.execute(action, context),
                timeout=action.timeout_seconds,
            )

            logger.debug(
                "action_executed",
                action_type=action.action_type.value,
                success=True,
            )

            return result

        except asyncio.TimeoutError:
            logger.error(
                "action_timeout",
                action_type=action.action_type.value,
                timeout=action.timeout_seconds,
            )
            raise TimeoutError(f"Action timed out after {action.timeout_seconds}s")

        except Exception as e:
            logger.error(
                "action_error",
                action_type=action.action_type.value,
                error=str(e),
            )
            raise

    def register_handler(
        self,
        action_type: ActionType,
        handler: "BaseActionHandler",
    ) -> None:
        """Register a custom action handler."""
        self._handlers[action_type] = handler
        logger.info("handler_registered", action_type=action_type.value)

    def get_handler(
        self,
        action_type: ActionType,
    ) -> Optional["BaseActionHandler"]:
        """Get a handler by action type."""
        return self._handlers.get(action_type)


class BaseActionHandler:
    """Base class for action handlers."""

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute the action."""
        raise NotImplementedError

    def resolve_params(
        self,
        params: Dict[str, Any],
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Resolve parameter expressions."""
        if not params:
            return {}

        resolved = {}
        for key, value in params.items():
            resolved[key] = context.resolve(value)
        return resolved


# === Built-in Handlers ===


class DelayActionHandler(BaseActionHandler):
    """Handler for delay actions."""

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute a delay action."""
        if action.delay_seconds:
            await asyncio.sleep(action.delay_seconds)
            return {
                "delayed": True,
                "seconds": action.delay_seconds,
            }

        if action.delay_until:
            # Parse datetime expression
            target = context.resolve(action.delay_until)
            if isinstance(target, str):
                target = datetime.fromisoformat(target)

            now = datetime.now()
            if target > now:
                delay = (target - now).total_seconds()
                await asyncio.sleep(delay)

            return {
                "delayed": True,
                "until": target.isoformat() if hasattr(target, 'isoformat') else str(target),
            }

        return {"delayed": False}


class TransformActionHandler(BaseActionHandler):
    """Handler for transform actions."""

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute a transform action."""
        if not action.transform_expression:
            return None

        # Resolve the expression
        result = context.resolve(action.transform_expression)

        # Store in output key if specified
        if action.transform_output_key:
            context.set(action.transform_output_key, result)

        return result


class LLMActionHandler(BaseActionHandler):
    """Handler for LLM completion actions."""

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute an LLM action."""
        prompt = context.resolve(action.llm_prompt or "")
        if not prompt:
            return {"error": "No prompt provided"}

        system_prompt = context.resolve(action.llm_system_prompt or "") if action.llm_system_prompt else None

        try:
            # Try to use Claude provider
            from aion.core.llm import LLMAdapter

            llm = LLMAdapter()

            response = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system_prompt,
                model=action.llm_model,
                temperature=action.llm_temperature,
                max_tokens=action.llm_max_tokens,
            )

            return {
                "response": response.get("content", "") if isinstance(response, dict) else str(response),
                "model": action.llm_model or "default",
            }

        except ImportError:
            logger.warning("llm_adapter_not_available")
            return {
                "response": f"[LLM simulation] Prompt: {prompt[:100]}...",
                "model": action.llm_model or "simulated",
                "simulated": True,
            }
        except Exception as e:
            logger.error("llm_action_error", error=str(e))
            return {
                "error": str(e),
                "model": action.llm_model,
            }


class ScriptActionHandler(BaseActionHandler):
    """Handler for script execution actions."""

    # Safe built-ins for script execution
    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "frozenset": frozenset,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute a script action."""
        if not action.script_code:
            return {"error": "No script code provided"}

        language = action.script_language.lower()

        if language == "python":
            return await self._execute_python(action.script_code, context)
        else:
            return {"error": f"Unsupported script language: {language}"}

    async def _execute_python(
        self,
        code: str,
        context: "ExecutionContext",
    ) -> Any:
        """Execute Python script in a sandbox."""
        # Create restricted globals
        restricted_globals = {
            "__builtins__": self.SAFE_BUILTINS,
            "context": context.to_safe_dict(),
            "inputs": context.get("inputs", {}),
            "trigger": context.get("trigger", {}),
            "steps": context.get("steps", {}),
        }

        # Local namespace for script
        local_namespace = {}

        try:
            # Execute the script
            exec(code, restricted_globals, local_namespace)

            # Look for a 'result' variable
            result = local_namespace.get("result")

            return {
                "result": result,
                "locals": {
                    k: v for k, v in local_namespace.items()
                    if not k.startswith("_") and not callable(v)
                },
            }

        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
            }
