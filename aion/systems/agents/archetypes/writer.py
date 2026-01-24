"""
AION Writer Agent

Specialist agent for content creation and editing.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from aion.systems.agents.types import AgentRole, AgentStatus, TeamTask
from aion.systems.agents.archetypes.base import BaseSpecialist

logger = structlog.get_logger(__name__)


class WriterAgent(BaseSpecialist):
    """
    Writer specialist agent.

    Capabilities:
    - Content creation
    - Editing and refinement
    - Summarization
    - Style adaptation
    - Documentation
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.WRITER

    async def process_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Process a writing task."""
        logger.info(
            "Processing writing task",
            agent_id=self.id[:8],
            task=task.title[:50],
        )

        self.update_status(AgentStatus.BUSY)

        try:
            # Determine task type
            task_text = f"{task.title} {task.description}".lower()

            if any(w in task_text for w in ["edit", "revise", "improve"]):
                result = await self._edit_task(task, context)
            elif any(w in task_text for w in ["summarize", "condense", "brief"]):
                result = await self._summarize_task(task, context)
            elif any(w in task_text for w in ["document", "documentation"]):
                result = await self._document_task(task, context)
            else:
                result = await self._write_task(task, context)

            self.instance.tasks_completed += 1
            return result

        except Exception as e:
            logger.error("Writing task failed", error=str(e))
            self.instance.tasks_failed += 1
            return {"success": False, "error": str(e)}

        finally:
            self.update_status(AgentStatus.IDLE)

    async def _write_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Write new content."""
        audience = context.get("audience", "general") if context else "general"
        tone = context.get("tone", "professional") if context else "professional"

        prompt = f"""Write content for this task:

## Task: {task.title}

## Topic/Requirements:
{task.description}

## Objective:
{task.objective}

## Audience: {audience}
## Tone: {tone}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Clear, engaging content"}

Create well-structured, engaging content that meets all requirements."""

        content = await self.think(prompt)

        return {
            "success": True,
            "type": "write",
            "content": content,
        }

    async def _edit_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Edit existing content."""
        original = context.get("content", task.description) if context else task.description

        prompt = f"""Edit and improve this content:

## Original Content:
{original}

## Editing Goals:
{task.objective or "Improve clarity, flow, and impact"}

Provide:
1. Edited content
2. Summary of changes
3. Rationale for major changes"""

        edited = await self.think(prompt)

        return {
            "success": True,
            "type": "edit",
            "edited_content": edited,
        }

    async def _summarize_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Summarize content."""
        original = context.get("content", task.description) if context else task.description
        length = context.get("length", "medium") if context else "medium"

        prompt = f"""Summarize this content:

## Original Content:
{original}

## Summary Length: {length}
## Focus: {task.objective or "Key points"}

Create a {length} summary that captures the essential information."""

        summary = await self.think(prompt)

        return {
            "success": True,
            "type": "summary",
            "summary": summary,
        }

    async def _document_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create documentation."""
        subject = context.get("subject", task.description) if context else task.description

        prompt = f"""Create documentation for:

## Subject:
{subject}

## Documentation Goals:
{task.objective or "Clear, comprehensive documentation"}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete, usable documentation"}

Create well-organized documentation with:
1. Overview/Introduction
2. Main content sections
3. Examples where helpful
4. Any necessary references"""

        docs = await self.think(prompt)

        return {
            "success": True,
            "type": "documentation",
            "documentation": docs,
        }

    async def write(
        self,
        topic: str,
        style: str = "professional",
        length: str = "medium",
    ) -> str:
        """Write content on a topic."""
        prompt = f"""Write {length} content about:

{topic}

Style: {style}
Length: {length}

Make it clear, engaging, and well-structured."""

        return await self.think(prompt)
