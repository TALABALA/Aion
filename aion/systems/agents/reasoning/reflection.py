"""
Self-Reflection System

Implements self-reflection and critique capabilities for agents,
enabling iterative improvement of outputs and reasoning.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

import structlog

logger = structlog.get_logger()


class CritiqueType(Enum):
    """Types of critique."""

    ACCURACY = "accuracy"  # Factual correctness
    COMPLETENESS = "completeness"  # Coverage of requirements
    CLARITY = "clarity"  # Understandability
    LOGIC = "logic"  # Logical consistency
    RELEVANCE = "relevance"  # Relevance to the task
    EFFICIENCY = "efficiency"  # Solution efficiency
    SAFETY = "safety"  # Safety and ethical concerns
    STYLE = "style"  # Style and presentation


@dataclass
class Critique:
    """A single critique point."""

    critique_type: CritiqueType
    issue: str
    severity: float  # 0-1, how severe the issue is
    suggestion: str
    location: Optional[str] = None  # Where in the output
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "critique_type": self.critique_type.value,
            "issue": self.issue,
            "severity": self.severity,
            "suggestion": self.suggestion,
            "location": self.location,
            "resolved": self.resolved,
        }


@dataclass
class ReflectionResult:
    """Result of a reflection cycle."""

    original_output: str
    critiques: list[Critique] = field(default_factory=list)
    revised_output: Optional[str] = None
    improvement_score: float = 0.0
    iterations: int = 0
    converged: bool = False
    reflection_chain: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_severity(self) -> float:
        """Total severity of all critiques."""
        return sum(c.severity for c in self.critiques)

    @property
    def unresolved_critiques(self) -> list[Critique]:
        """Get unresolved critiques."""
        return [c for c in self.critiques if not c.resolved]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_output": self.original_output,
            "critiques": [c.to_dict() for c in self.critiques],
            "revised_output": self.revised_output,
            "improvement_score": self.improvement_score,
            "iterations": self.iterations,
            "converged": self.converged,
            "reflection_chain": self.reflection_chain,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReflectionConfig:
    """Configuration for self-reflection."""

    # Iteration limits
    max_iterations: int = 3
    min_iterations: int = 1

    # Convergence
    convergence_threshold: float = 0.1  # Stop if improvement < this
    severity_threshold: float = 0.2  # Only address issues above this

    # Critique types to use
    critique_types: list[CritiqueType] = field(default_factory=lambda: list(CritiqueType))

    # Prompts
    critique_prompt_template: str = """Review the following output and provide critique.

Task: {task}

Output to review:
{output}

Provide specific critiques in the following format:
- [TYPE] Issue: <description> | Severity: <0-1> | Suggestion: <improvement>

Types: accuracy, completeness, clarity, logic, relevance, efficiency, safety, style

Critiques:"""

    revision_prompt_template: str = """Revise the output based on the following critiques.

Original task: {task}

Original output:
{output}

Critiques to address:
{critiques}

Provide an improved version that addresses these issues:"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "min_iterations": self.min_iterations,
            "convergence_threshold": self.convergence_threshold,
            "severity_threshold": self.severity_threshold,
            "critique_types": [ct.value for ct in self.critique_types],
        }


# Type for LLM generation function
GenerateFn = Callable[[str], Awaitable[str]]


class SelfReflection:
    """
    Self-reflection system for iterative output improvement.

    Features:
    - Multi-dimensional critique using Llama 3.3 70B
    - Iterative refinement
    - Convergence detection
    - Critique severity weighting
    - Reflection chains for transparency
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[ReflectionConfig] = None,
        generate_fn: Optional[GenerateFn] = None,
    ):
        self.agent_id = agent_id or "default"
        self.config = config or ReflectionConfig()
        self._custom_generate_fn = generate_fn
        self._llm_provider = None

        self._reflection_count = 0

    async def _get_llm_provider(self):
        """Get or create LLM provider for reflection."""
        if self._llm_provider is None:
            from aion.systems.agents.llm_integration import SOTALLMProvider
            self._llm_provider = await SOTALLMProvider.get_instance()
        return self._llm_provider

    @property
    def generate_fn(self):
        """Backward compatibility property."""
        return self._custom_generate_fn

    async def reflect(
        self,
        task: str,
        output: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ReflectionResult:
        """
        Perform self-reflection on an output.

        Args:
            task: The original task description
            output: The output to reflect on
            context: Optional additional context

        Returns:
            ReflectionResult with critiques and revised output
        """
        self._reflection_count += 1

        result = ReflectionResult(original_output=output)
        current_output = output

        logger.info("reflection_started", task_length=len(task), output_length=len(output))

        for iteration in range(self.config.max_iterations):
            result.iterations = iteration + 1

            # Generate critiques
            critiques = await self._generate_critiques(task, current_output)
            result.critiques.extend(critiques)

            # Add to reflection chain
            critique_summary = f"Iteration {iteration + 1}: Found {len(critiques)} issues"
            result.reflection_chain.append(critique_summary)

            # Filter by severity
            significant_critiques = [
                c for c in critiques
                if c.severity >= self.config.severity_threshold and not c.resolved
            ]

            # Check convergence
            if not significant_critiques and iteration >= self.config.min_iterations - 1:
                result.converged = True
                break

            # Generate revision
            if significant_critiques:
                revised = await self._generate_revision(
                    task, current_output, significant_critiques
                )
                if revised:
                    # Mark addressed critiques as resolved
                    for c in significant_critiques:
                        c.resolved = True

                    current_output = revised
                    result.reflection_chain.append(f"Revised output (length: {len(revised)})")

        result.revised_output = current_output

        # Calculate improvement score
        result.improvement_score = self._calculate_improvement(result)

        logger.info(
            "reflection_completed",
            iterations=result.iterations,
            critiques=len(result.critiques),
            improvement=result.improvement_score,
            converged=result.converged,
        )

        return result

    async def _generate_critiques(
        self,
        task: str,
        output: str,
    ) -> list[Critique]:
        """Generate critiques for an output using Llama 3.3 70B."""
        # Use custom function if provided
        if self._custom_generate_fn:
            prompt = self.config.critique_prompt_template.format(
                task=task,
                output=output,
            )
            response = await self._custom_generate_fn(prompt)
            return self._parse_critiques(response)

        # Use real LLM provider
        try:
            llm_provider = await self._get_llm_provider()
            reflection = await llm_provider.reflect_on_output(task, output)

            critiques = []
            # Convert weaknesses to critiques
            for weakness in reflection.get("weaknesses", []):
                critiques.append(Critique(
                    critique_type=CritiqueType.LOGIC,
                    issue=weakness,
                    severity=0.6,
                    suggestion="Address this issue",
                ))

            # Add suggestions as critiques
            for suggestion in reflection.get("suggestions", []):
                critiques.append(Critique(
                    critique_type=CritiqueType.COMPLETENESS,
                    issue=f"Improvement needed: {suggestion}",
                    severity=0.4,
                    suggestion=suggestion,
                ))

            return critiques

        except Exception as e:
            logger.warning("critique_generation_fallback", error=str(e))
            return self._heuristic_critiques(output)

    async def _generate_revision(
        self,
        task: str,
        output: str,
        critiques: list[Critique],
    ) -> Optional[str]:
        """Generate a revised output addressing critiques using Llama 3.3 70B."""
        # Use custom function if provided
        if self._custom_generate_fn:
            critiques_text = "\n".join(
                f"- [{c.critique_type.value.upper()}] {c.issue} (Severity: {c.severity:.1f})\n  Suggestion: {c.suggestion}"
                for c in critiques
            )
            prompt = self.config.revision_prompt_template.format(
                task=task,
                output=output,
                critiques=critiques_text,
            )
            return await self._custom_generate_fn(prompt)

        # Use real LLM provider
        try:
            llm_provider = await self._get_llm_provider()
            improved, _ = await llm_provider.critique_and_improve(task, output, max_iterations=1)
            return improved
        except Exception as e:
            logger.warning("revision_generation_fallback", error=str(e))
            return None

    def _parse_critiques(self, response: str) -> list[Critique]:
        """Parse critiques from LLM response."""
        critiques = []

        for line in response.split("\n"):
            line = line.strip()
            if not line or not line.startswith("-"):
                continue

            try:
                # Parse format: - [TYPE] Issue: <desc> | Severity: <num> | Suggestion: <sugg>
                line = line.lstrip("- ")

                # Extract type
                if "[" in line and "]" in line:
                    type_start = line.index("[")
                    type_end = line.index("]")
                    type_str = line[type_start + 1:type_end].lower()
                    line = line[type_end + 1:].strip()

                    try:
                        critique_type = CritiqueType(type_str)
                    except ValueError:
                        critique_type = CritiqueType.LOGIC
                else:
                    critique_type = CritiqueType.LOGIC

                # Parse components
                parts = line.split("|")
                issue = parts[0].replace("Issue:", "").strip() if parts else line
                severity = 0.5
                suggestion = ""

                for part in parts[1:]:
                    part = part.strip()
                    if part.lower().startswith("severity"):
                        try:
                            sev_str = part.split(":")[1].strip()
                            severity = float(sev_str)
                        except (IndexError, ValueError):
                            pass
                    elif part.lower().startswith("suggestion"):
                        suggestion = part.split(":", 1)[1].strip() if ":" in part else part

                critiques.append(Critique(
                    critique_type=critique_type,
                    issue=issue,
                    severity=severity,
                    suggestion=suggestion,
                ))

            except Exception as e:
                logger.debug("critique_parse_error", line=line, error=str(e))
                continue

        return critiques

    def _heuristic_critiques(self, output: str) -> list[Critique]:
        """Generate basic heuristic critiques."""
        critiques = []

        # Length check
        if len(output) < 50:
            critiques.append(Critique(
                critique_type=CritiqueType.COMPLETENESS,
                issue="Output may be too brief",
                severity=0.4,
                suggestion="Consider expanding with more detail",
            ))

        # Check for common issues
        if output.count("TODO") > 0:
            critiques.append(Critique(
                critique_type=CritiqueType.COMPLETENESS,
                issue="Output contains TODO items",
                severity=0.6,
                suggestion="Complete the TODO items",
            ))

        if "I don't know" in output or "I'm not sure" in output:
            critiques.append(Critique(
                critique_type=CritiqueType.ACCURACY,
                issue="Output expresses uncertainty",
                severity=0.5,
                suggestion="Research and provide more definitive information",
            ))

        return critiques

    def _calculate_improvement(self, result: ReflectionResult) -> float:
        """Calculate improvement score."""
        if not result.critiques:
            return 1.0

        resolved = sum(1 for c in result.critiques if c.resolved)
        total = len(result.critiques)

        # Base score from resolution rate
        resolution_rate = resolved / total if total > 0 else 1.0

        # Adjust for severity
        total_severity = sum(c.severity for c in result.critiques)
        resolved_severity = sum(c.severity for c in result.critiques if c.resolved)
        severity_resolution = resolved_severity / total_severity if total_severity > 0 else 1.0

        # Combined score
        return (resolution_rate + severity_resolution) / 2

    async def critique_only(
        self,
        task: str,
        output: str,
    ) -> list[Critique]:
        """Generate critiques without revision."""
        return await self._generate_critiques(task, output)

    async def compare_outputs(
        self,
        task: str,
        output_a: str,
        output_b: str,
    ) -> dict[str, Any]:
        """Compare two outputs and determine which is better."""
        # Get critiques for both
        critiques_a = await self._generate_critiques(task, output_a)
        critiques_b = await self._generate_critiques(task, output_b)

        # Calculate scores
        score_a = 1.0 - sum(c.severity for c in critiques_a) / max(1, len(critiques_a))
        score_b = 1.0 - sum(c.severity for c in critiques_b) / max(1, len(critiques_b))

        return {
            "output_a_score": score_a,
            "output_b_score": score_b,
            "winner": "A" if score_a > score_b else "B" if score_b > score_a else "tie",
            "output_a_critiques": len(critiques_a),
            "output_b_critiques": len(critiques_b),
            "critiques_a": [c.to_dict() for c in critiques_a],
            "critiques_b": [c.to_dict() for c in critiques_b],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get reflection statistics."""
        return {
            "total_reflections": self._reflection_count,
            "config": self.config.to_dict(),
        }
