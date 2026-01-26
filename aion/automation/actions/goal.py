"""
AION Goal Action Handler

Create and manage goals from workflows.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import structlog

from aion.automation.types import ActionConfig, GoalOperation
from aion.automation.actions.executor import BaseActionHandler

if TYPE_CHECKING:
    from aion.automation.execution.context import ExecutionContext

logger = structlog.get_logger(__name__)


class GoalActionHandler(BaseActionHandler):
    """
    Handler for goal operations.

    Integrates with AION's autonomous goal system.
    """

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute a goal action."""
        operation = action.goal_operation
        if not operation:
            return {"error": "No goal operation specified"}

        if operation == "create":
            return await self._create_goal(action, context)
        elif operation == "pause":
            return await self._pause_goal(action, context)
        elif operation == "resume":
            return await self._resume_goal(action, context)
        elif operation == "abandon":
            return await self._abandon_goal(action, context)
        elif operation == "update":
            return await self._update_goal(action, context)
        elif operation == "status":
            return await self._get_goal_status(action, context)
        else:
            return {"error": f"Unknown goal operation: {operation}"}

    async def _create_goal(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Create a new goal."""
        title = context.resolve(action.goal_title or "")
        description = context.resolve(action.goal_description or "")
        config = self.resolve_params(action.goal_config or {}, context)

        if not title:
            return {"error": "Goal title is required", "success": False}

        logger.info("creating_goal", title=title)

        try:
            from aion.systems.goals.manager import AutonomousGoalManager
            from aion.systems.goals.types import Goal, GoalPriority

            manager = AutonomousGoalManager()
            await manager.initialize()

            goal = Goal(
                title=title,
                description=description,
                priority=GoalPriority(config.get("priority", "medium")),
                metadata={
                    "workflow_id": context.execution.workflow_id,
                    "execution_id": context.execution.id,
                    **config.get("metadata", {}),
                },
            )

            goal_id = await manager.submit_goal(goal)

            return {
                "operation": "create",
                "goal_id": goal_id,
                "title": title,
                "success": True,
            }

        except ImportError:
            logger.warning("goal_manager_not_available")
            import uuid
            simulated_id = f"goal-{uuid.uuid4().hex[:8]}"
            return {
                "operation": "create",
                "goal_id": simulated_id,
                "title": title,
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            logger.error("goal_create_error", error=str(e))
            return {
                "operation": "create",
                "error": str(e),
                "success": False,
            }

    async def _pause_goal(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Pause a goal."""
        config = action.goal_config or {}
        goal_id = context.resolve(action.goal_id or config.get("goal_id", ""))

        if not goal_id:
            return {"error": "No goal_id specified", "success": False}

        logger.info("pausing_goal", goal_id=goal_id)

        try:
            from aion.systems.goals.manager import AutonomousGoalManager

            manager = AutonomousGoalManager()
            await manager.initialize()
            await manager.pause_goal(goal_id)

            return {
                "operation": "pause",
                "goal_id": goal_id,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "pause",
                "goal_id": goal_id,
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            return {
                "operation": "pause",
                "goal_id": goal_id,
                "error": str(e),
                "success": False,
            }

    async def _resume_goal(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Resume a paused goal."""
        config = action.goal_config or {}
        goal_id = context.resolve(action.goal_id or config.get("goal_id", ""))

        if not goal_id:
            return {"error": "No goal_id specified", "success": False}

        logger.info("resuming_goal", goal_id=goal_id)

        try:
            from aion.systems.goals.manager import AutonomousGoalManager

            manager = AutonomousGoalManager()
            await manager.initialize()
            await manager.resume_goal(goal_id)

            return {
                "operation": "resume",
                "goal_id": goal_id,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "resume",
                "goal_id": goal_id,
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            return {
                "operation": "resume",
                "goal_id": goal_id,
                "error": str(e),
                "success": False,
            }

    async def _abandon_goal(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Abandon a goal."""
        config = action.goal_config or {}
        goal_id = context.resolve(action.goal_id or config.get("goal_id", ""))
        reason = context.resolve(config.get("reason", "Abandoned by workflow"))

        if not goal_id:
            return {"error": "No goal_id specified", "success": False}

        logger.info("abandoning_goal", goal_id=goal_id, reason=reason)

        try:
            from aion.systems.goals.manager import AutonomousGoalManager

            manager = AutonomousGoalManager()
            await manager.initialize()
            await manager.abandon_goal(goal_id, reason=reason)

            return {
                "operation": "abandon",
                "goal_id": goal_id,
                "reason": reason,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "abandon",
                "goal_id": goal_id,
                "reason": reason,
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            return {
                "operation": "abandon",
                "goal_id": goal_id,
                "error": str(e),
                "success": False,
            }

    async def _update_goal(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Update a goal."""
        config = action.goal_config or {}
        goal_id = context.resolve(action.goal_id or config.get("goal_id", ""))
        updates = self.resolve_params(config.get("updates", {}), context)

        if not goal_id:
            return {"error": "No goal_id specified", "success": False}

        logger.info("updating_goal", goal_id=goal_id)

        try:
            from aion.systems.goals.manager import AutonomousGoalManager

            manager = AutonomousGoalManager()
            await manager.initialize()
            await manager.update_goal(goal_id, **updates)

            return {
                "operation": "update",
                "goal_id": goal_id,
                "updates": updates,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "update",
                "goal_id": goal_id,
                "updates": updates,
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            return {
                "operation": "update",
                "goal_id": goal_id,
                "error": str(e),
                "success": False,
            }

    async def _get_goal_status(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Get goal status."""
        config = action.goal_config or {}
        goal_id = context.resolve(action.goal_id or config.get("goal_id", ""))

        if not goal_id:
            return {"error": "No goal_id specified", "success": False}

        try:
            from aion.systems.goals.manager import AutonomousGoalManager

            manager = AutonomousGoalManager()
            await manager.initialize()
            goal = await manager.get_goal(goal_id)

            if not goal:
                return {
                    "operation": "status",
                    "goal_id": goal_id,
                    "error": "Goal not found",
                    "success": False,
                }

            return {
                "operation": "status",
                "goal_id": goal_id,
                "title": goal.title,
                "status": goal.status.value,
                "progress": goal.progress,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "status",
                "goal_id": goal_id,
                "status": "active",
                "progress": 0.5,
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            return {
                "operation": "status",
                "goal_id": goal_id,
                "error": str(e),
                "success": False,
            }
