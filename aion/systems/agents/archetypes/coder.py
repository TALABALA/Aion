"""
AION Coder Agent

Specialist agent for code generation and review.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from aion.systems.agents.types import AgentRole, AgentStatus, TeamTask
from aion.systems.agents.archetypes.base import BaseSpecialist

logger = structlog.get_logger(__name__)


class CoderAgent(BaseSpecialist):
    """
    Coder specialist agent.

    Capabilities:
    - Code generation
    - Debugging and troubleshooting
    - Code review
    - Refactoring
    - Test writing
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.CODER

    async def process_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Process a coding task."""
        logger.info(
            "Processing coding task",
            agent_id=self.id[:8],
            task=task.title[:50],
        )

        self.update_status(AgentStatus.BUSY)

        try:
            # Determine task type
            task_text = f"{task.title} {task.description}".lower()

            if any(w in task_text for w in ["debug", "fix", "bug", "error"]):
                result = await self._debug_task(task, context)
            elif any(w in task_text for w in ["review", "check", "audit"]):
                result = await self._review_task(task, context)
            elif any(w in task_text for w in ["refactor", "improve", "optimize"]):
                result = await self._refactor_task(task, context)
            elif any(w in task_text for w in ["test", "testing"]):
                result = await self._test_task(task, context)
            else:
                result = await self._implement_task(task, context)

            self.instance.tasks_completed += 1
            return result

        except Exception as e:
            logger.error("Coding task failed", error=str(e))
            self.instance.tasks_failed += 1
            return {"success": False, "error": str(e)}

        finally:
            self.update_status(AgentStatus.IDLE)

    async def _implement_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Implement code for a task."""
        prompt = f"""Implement code for this task:

## Task: {task.title}

## Requirements:
{task.description}

## Objective:
{task.objective}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Working, tested code"}

Provide:
1. Complete, working code
2. Clear comments
3. Usage examples
4. Any assumptions made"""

        code = await self.think(prompt)

        return {
            "success": True,
            "type": "implementation",
            "code": code,
        }

    async def _debug_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Debug code."""
        prompt = f"""Debug this issue:

## Problem: {task.title}

## Description:
{task.description}

## Context:
{context.get('code', 'No code provided') if context else 'No context'}
{context.get('error', 'No error message') if context else ''}

Provide:
1. Root cause analysis
2. The fix
3. Explanation of what was wrong
4. Prevention suggestions"""

        analysis = await self.think(prompt)

        return {
            "success": True,
            "type": "debug",
            "analysis": analysis,
        }

    async def _review_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Review code."""
        code = context.get("code", task.description) if context else task.description

        prompt = f"""Review this code:

## Code:
{code}

## Review Focus:
{task.objective or "General code review"}

Provide:
1. Overall assessment
2. Issues found (bugs, security, performance)
3. Suggestions for improvement
4. Positive aspects
5. Final recommendation"""

        review = await self.think(prompt)

        return {
            "success": True,
            "type": "review",
            "review": review,
        }

    async def _refactor_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Refactor code."""
        code = context.get("code", task.description) if context else task.description

        prompt = f"""Refactor this code:

## Current Code:
{code}

## Refactoring Goals:
{task.objective or "Improve code quality and maintainability"}

Provide:
1. Refactored code
2. Changes made and why
3. Benefits of the changes
4. Any trade-offs"""

        refactored = await self.think(prompt)

        return {
            "success": True,
            "type": "refactor",
            "refactored_code": refactored,
        }

    async def _test_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Write tests."""
        code = context.get("code", "") if context else ""

        prompt = f"""Write tests for:

## Task: {task.title}

## Code to Test:
{code if code else task.description}

## Testing Requirements:
{task.objective or "Comprehensive test coverage"}

Provide:
1. Test cases covering main functionality
2. Edge cases
3. Error scenarios
4. Test framework setup if needed"""

        tests = await self.think(prompt)

        return {
            "success": True,
            "type": "tests",
            "tests": tests,
        }

    async def generate_code(
        self,
        specification: str,
        language: str = "python",
    ) -> str:
        """Generate code from a specification."""
        prompt = f"""Generate {language} code for:

{specification}

Requirements:
1. Clean, idiomatic code
2. Proper error handling
3. Clear documentation
4. Follow best practices"""

        return await self.think(prompt)
