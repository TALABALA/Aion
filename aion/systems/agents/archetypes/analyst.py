"""
AION Analyst Agent

Specialist agent for data analysis and insights.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from aion.systems.agents.types import AgentRole, AgentStatus, TeamTask
from aion.systems.agents.archetypes.base import BaseSpecialist

logger = structlog.get_logger(__name__)


class AnalystAgent(BaseSpecialist):
    """
    Analyst specialist agent.

    Capabilities:
    - Data analysis
    - Pattern recognition
    - Statistical analysis
    - Visualization recommendations
    - Insight generation
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.ANALYST

    async def process_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Process an analysis task."""
        logger.info(
            "Processing analysis task",
            agent_id=self.id[:8],
            task=task.title[:50],
        )

        self.update_status(AgentStatus.BUSY)

        try:
            # Phase 1: Understand the data and questions
            understanding = await self._understand_data(task, context)

            # Phase 2: Perform analysis
            analysis = await self._perform_analysis(task, understanding, context)

            # Phase 3: Generate insights
            insights = await self._generate_insights(task, analysis)

            # Phase 4: Create recommendations
            recommendations = await self._create_recommendations(task, insights)

            result = {
                "success": True,
                "understanding": understanding,
                "analysis": analysis,
                "insights": insights,
                "recommendations": recommendations,
            }

            self.instance.tasks_completed += 1
            return result

        except Exception as e:
            logger.error("Analysis task failed", error=str(e))
            self.instance.tasks_failed += 1
            return {"success": False, "error": str(e)}

        finally:
            self.update_status(AgentStatus.IDLE)

    async def _understand_data(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> str:
        """Understand the data and analysis requirements."""
        data_desc = context.get("data", task.description) if context else task.description

        prompt = f"""Understand this data analysis task:

## Task: {task.title}

## Data/Context:
{data_desc}

## Questions to Answer:
{task.objective}

Provide:
1. What data is being analyzed
2. Key questions to answer
3. Relevant metrics
4. Analysis approach"""

        return await self.think(prompt)

    async def _perform_analysis(
        self,
        task: TeamTask,
        understanding: str,
        context: Optional[dict[str, Any]],
    ) -> str:
        """Perform the analysis."""
        data_desc = context.get("data", task.description) if context else task.description

        prompt = f"""Perform analysis on this data:

## Task: {task.title}

## Understanding:
{understanding}

## Data:
{data_desc}

Provide detailed analysis:
1. Statistical summaries
2. Patterns identified
3. Correlations found
4. Anomalies detected
5. Trends observed"""

        return await self.think(prompt)

    async def _generate_insights(
        self,
        task: TeamTask,
        analysis: str,
    ) -> str:
        """Generate insights from analysis."""
        prompt = f"""Generate insights from this analysis:

## Task: {task.title}

## Analysis Results:
{analysis}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Actionable insights"}

Provide:
1. Key findings
2. Surprising discoveries
3. Implications
4. Confidence levels"""

        return await self.think(prompt)

    async def _create_recommendations(
        self,
        task: TeamTask,
        insights: str,
    ) -> str:
        """Create recommendations based on insights."""
        prompt = f"""Create recommendations based on these insights:

## Task: {task.title}

## Insights:
{insights}

Provide:
1. Actionable recommendations
2. Prioritization
3. Expected impact
4. Implementation considerations"""

        return await self.think(prompt)

    async def analyze(
        self,
        data: str,
        questions: list[str],
    ) -> str:
        """Analyze data to answer specific questions."""
        questions_str = "\n".join(f"- {q}" for q in questions)

        prompt = f"""Analyze this data to answer the following questions:

## Data:
{data}

## Questions:
{questions_str}

Provide thorough answers with supporting evidence from the data."""

        return await self.think(prompt)
