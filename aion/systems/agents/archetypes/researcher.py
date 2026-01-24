"""
AION Researcher Agent

Specialist agent for information gathering and research.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from aion.systems.agents.types import AgentRole, TeamTask
from aion.systems.agents.archetypes.base import BaseSpecialist

logger = structlog.get_logger(__name__)


class ResearcherAgent(BaseSpecialist):
    """
    Researcher specialist agent.

    Capabilities:
    - Information gathering from multiple sources
    - Source evaluation and credibility assessment
    - Synthesis and summarization
    - Gap identification
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.RESEARCHER

    async def process_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Process a research task."""
        logger.info(
            "Processing research task",
            agent_id=self.id[:8],
            task=task.title[:50],
        )

        self.update_status(AgentStatus.BUSY)

        try:
            # Phase 1: Understand the research question
            research_plan = await self._create_research_plan(task)

            # Phase 2: Gather information
            findings = await self._gather_information(task, research_plan)

            # Phase 3: Synthesize findings
            synthesis = await self._synthesize_findings(task, findings)

            # Phase 4: Identify gaps and limitations
            assessment = await self._assess_completeness(task, synthesis)

            result = {
                "success": True,
                "research_plan": research_plan,
                "findings": findings,
                "synthesis": synthesis,
                "assessment": assessment,
            }

            self.instance.tasks_completed += 1
            return result

        except Exception as e:
            logger.error("Research task failed", error=str(e))
            self.instance.tasks_failed += 1
            return {"success": False, "error": str(e)}

        finally:
            self.update_status(AgentStatus.IDLE)

    async def _create_research_plan(self, task: TeamTask) -> str:
        """Create a research plan."""
        prompt = f"""Create a research plan for:

Topic: {task.title}
Objective: {task.objective}

Outline:
1. Key questions to answer
2. Information sources to consult
3. Search strategies
4. Expected deliverables

Be systematic and thorough."""

        return await self.think(prompt)

    async def _gather_information(
        self,
        task: TeamTask,
        research_plan: str,
    ) -> str:
        """Gather information based on research plan."""
        prompt = f"""Based on this research plan, gather and organize relevant information:

Topic: {task.title}
Research Plan:
{research_plan}

Provide:
1. Key facts and data
2. Different perspectives
3. Source references
4. Confidence levels

Be thorough and cite sources where possible."""

        return await self.think(prompt)

    async def _synthesize_findings(
        self,
        task: TeamTask,
        findings: str,
    ) -> str:
        """Synthesize research findings."""
        prompt = f"""Synthesize these research findings into a coherent summary:

Topic: {task.title}
Objective: {task.objective}

Findings:
{findings}

Create a clear synthesis that:
1. Answers the key questions
2. Highlights main insights
3. Notes patterns and themes
4. Provides actionable conclusions"""

        return await self.think(prompt)

    async def _assess_completeness(
        self,
        task: TeamTask,
        synthesis: str,
    ) -> str:
        """Assess research completeness."""
        success_criteria = "\n".join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete the research"

        prompt = f"""Assess the completeness of this research:

Success Criteria:
{success_criteria}

Synthesis:
{synthesis}

Evaluate:
1. What criteria are met?
2. What gaps remain?
3. What limitations exist?
4. Confidence level overall?"""

        return await self.think(prompt)

    async def search(self, query: str) -> str:
        """Perform a research search."""
        prompt = f"""Research this query: {query}

Provide comprehensive findings with:
1. Key information
2. Multiple perspectives
3. Source quality assessment
4. Confidence level"""

        return await self.think(prompt)


from aion.systems.agents.types import AgentStatus  # Import at end
