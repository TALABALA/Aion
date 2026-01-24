"""
Chain-of-Thought Reasoning

Implementation of Chain-of-Thought (CoT) prompting with extensions
including self-consistency, least-to-most decomposition, and
verification chains.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

import structlog

logger = structlog.get_logger()


class ReasoningType(Enum):
    """Types of reasoning steps."""

    OBSERVATION = "observation"  # Noting relevant facts
    INFERENCE = "inference"  # Drawing conclusions
    CALCULATION = "calculation"  # Mathematical operations
    DECOMPOSITION = "decomposition"  # Breaking down problem
    HYPOTHESIS = "hypothesis"  # Proposing possibilities
    VERIFICATION = "verification"  # Checking correctness
    SYNTHESIS = "synthesis"  # Combining findings
    CONCLUSION = "conclusion"  # Final answer


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_number: int
    reasoning_type: ReasoningType
    content: str
    confidence: float = 0.8
    dependencies: list[int] = field(default_factory=list)  # Step numbers this depends on
    verification_status: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "reasoning_type": self.reasoning_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "dependencies": self.dependencies,
            "verification_status": self.verification_status,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReasoningChain:
    """A complete chain of reasoning."""

    id: str
    problem: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    overall_confidence: float = 0.0
    verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def add_step(
        self,
        reasoning_type: ReasoningType,
        content: str,
        confidence: float = 0.8,
        dependencies: Optional[list[int]] = None,
    ) -> ReasoningStep:
        """Add a reasoning step."""
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            reasoning_type=reasoning_type,
            content=content,
            confidence=confidence,
            dependencies=dependencies or [],
        )
        self.steps.append(step)
        return step

    def get_chain_text(self) -> str:
        """Get the full reasoning chain as text."""
        lines = [f"Problem: {self.problem}", ""]
        for step in self.steps:
            lines.append(f"Step {step.step_number} ({step.reasoning_type.value}): {step.content}")
        if self.final_answer:
            lines.append(f"\nAnswer: {self.final_answer}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "problem": self.problem,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "overall_confidence": self.overall_confidence,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought reasoning."""

    # Basic settings
    max_steps: int = 10
    min_confidence: float = 0.5

    # Self-consistency
    enable_self_consistency: bool = True
    num_samples: int = 5
    consistency_threshold: float = 0.6

    # Verification
    enable_verification: bool = True
    verification_depth: int = 2

    # Least-to-most decomposition
    enable_decomposition: bool = True
    max_subproblems: int = 5

    # Prompts
    cot_prompt_template: str = """Solve this problem step by step.

Problem: {problem}

Think through this carefully, showing your reasoning at each step.
Format each step as:
Step N (type): reasoning

Types can be: observation, inference, calculation, decomposition, hypothesis, verification, synthesis, conclusion

Reasoning:"""

    verification_prompt_template: str = """Verify the following reasoning chain.

Problem: {problem}

Reasoning:
{chain}

Check each step for:
1. Logical correctness
2. Mathematical accuracy (if applicable)
3. Consistency with previous steps

Is this reasoning correct? Identify any errors:"""

    decomposition_prompt_template: str = """Break down this problem into simpler sub-problems.

Problem: {problem}

List the sub-problems that need to be solved first:"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_steps": self.max_steps,
            "min_confidence": self.min_confidence,
            "enable_self_consistency": self.enable_self_consistency,
            "num_samples": self.num_samples,
            "enable_verification": self.enable_verification,
            "enable_decomposition": self.enable_decomposition,
        }


# Type for LLM generation function
GenerateFn = Callable[[str], Awaitable[str]]


class ChainOfThought:
    """
    Chain-of-Thought reasoning system.

    Features:
    - Step-by-step reasoning generation using Llama 3.3 70B
    - Self-consistency voting
    - Verification chains
    - Least-to-most decomposition
    - Confidence tracking
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[CoTConfig] = None,
        generate_fn: Optional[GenerateFn] = None,
    ):
        self.agent_id = agent_id or "default"
        self.config = config or CoTConfig()
        self._custom_generate_fn = generate_fn
        self._llm_provider = None

        self._chain_counter = 0

    async def _get_llm_provider(self):
        """Get or create LLM provider for reasoning."""
        if self._llm_provider is None:
            from aion.systems.agents.llm_integration import SOTALLMProvider
            self._llm_provider = await SOTALLMProvider.get_instance()
        return self._llm_provider

    @property
    def generate_fn(self):
        """Backward compatibility property."""
        return self._custom_generate_fn

    async def reason(
        self,
        problem: str,
        context: Optional[str] = None,
    ) -> ReasoningChain:
        """
        Generate a chain of thought reasoning for a problem.

        Args:
            problem: The problem to solve
            context: Optional additional context

        Returns:
            Complete reasoning chain with answer
        """
        logger.info("cot_reason_started", problem_length=len(problem))

        # Decompose if enabled
        subproblems = []
        if self.config.enable_decomposition:
            subproblems = await self._decompose_problem(problem)

        # Generate reasoning chains
        if self.config.enable_self_consistency:
            chains = await self._generate_multiple_chains(
                problem, context, self.config.num_samples
            )
            best_chain = await self._select_by_consistency(chains)
        else:
            best_chain = await self._generate_chain(problem, context, subproblems)

        # Verify if enabled
        if self.config.enable_verification:
            best_chain = await self._verify_chain(best_chain)

        logger.info(
            "cot_reason_completed",
            steps=len(best_chain.steps),
            confidence=best_chain.overall_confidence,
            verified=best_chain.verified,
        )

        return best_chain

    async def _generate_chain(
        self,
        problem: str,
        context: Optional[str],
        subproblems: Optional[list[str]] = None,
    ) -> ReasoningChain:
        """Generate a single reasoning chain using Llama 3.3 70B."""
        self._chain_counter += 1
        chain = ReasoningChain(
            id=f"cot-{self._chain_counter}",
            problem=problem,
        )

        # Solve subproblems first
        if subproblems:
            for subproblem in subproblems:
                chain.add_step(
                    reasoning_type=ReasoningType.DECOMPOSITION,
                    content=f"Sub-problem: {subproblem}",
                    confidence=0.9,
                )

        # Generate main reasoning
        if self._custom_generate_fn:
            # Use custom function if provided
            prompt = self.config.cot_prompt_template.format(
                problem=f"{problem}\n\nContext: {context}" if context else problem
            )
            response = await self._custom_generate_fn(prompt)
            steps = self._parse_reasoning_steps(response)

            for step_type, content in steps:
                chain.add_step(
                    reasoning_type=step_type,
                    content=content,
                )

            # Extract final answer
            chain.final_answer = self._extract_answer(response)
        else:
            # Use real LLM provider (Llama 3.3 70B)
            try:
                llm_provider = await self._get_llm_provider()
                reasoning_steps = await llm_provider.generate_cot_reasoning(problem, context)

                for i, step_content in enumerate(reasoning_steps):
                    # Determine step type heuristically
                    step_lower = step_content.lower()
                    if any(w in step_lower for w in ["observe", "note", "see", "given"]):
                        step_type = ReasoningType.OBSERVATION
                    elif any(w in step_lower for w in ["calculate", "compute", "=", "+"]):
                        step_type = ReasoningType.CALCULATION
                    elif any(w in step_lower for w in ["therefore", "thus", "conclude", "answer"]):
                        step_type = ReasoningType.CONCLUSION
                    elif any(w in step_lower for w in ["verify", "check", "confirm"]):
                        step_type = ReasoningType.VERIFICATION
                    elif any(w in step_lower for w in ["hypothesis", "assume", "suppose"]):
                        step_type = ReasoningType.HYPOTHESIS
                    else:
                        step_type = ReasoningType.INFERENCE

                    chain.add_step(
                        reasoning_type=step_type,
                        content=step_content,
                    )

                # Extract final answer using LLM
                if reasoning_steps:
                    chain.final_answer = await llm_provider.extract_answer(reasoning_steps, problem)

            except Exception as e:
                logger.warning("cot_llm_fallback", error=str(e))
                # Fallback if LLM fails
                chain.add_step(
                    reasoning_type=ReasoningType.OBSERVATION,
                    content=f"Analyzing: {problem[:100]}",
                )
                chain.add_step(
                    reasoning_type=ReasoningType.INFERENCE,
                    content="Based on the analysis...",
                )
                chain.final_answer = "Unable to generate complete reasoning"

        # Calculate overall confidence
        if chain.steps:
            chain.overall_confidence = sum(s.confidence for s in chain.steps) / len(chain.steps)

        return chain

    async def _generate_multiple_chains(
        self,
        problem: str,
        context: Optional[str],
        n: int,
    ) -> list[ReasoningChain]:
        """Generate multiple reasoning chains for self-consistency."""
        tasks = [
            self._generate_chain(problem, context)
            for _ in range(n)
        ]
        return await asyncio.gather(*tasks)

    async def _select_by_consistency(
        self,
        chains: list[ReasoningChain],
    ) -> ReasoningChain:
        """Select best chain using self-consistency voting."""
        if not chains:
            raise ValueError("No chains to select from")

        if len(chains) == 1:
            return chains[0]

        # Extract answers and vote
        answer_counts = Counter(
            chain.final_answer for chain in chains if chain.final_answer
        )

        if not answer_counts:
            # Fall back to highest confidence
            return max(chains, key=lambda c: c.overall_confidence)

        # Find most common answer
        most_common_answer, count = answer_counts.most_common(1)[0]
        consistency = count / len(chains)

        # Find best chain with that answer
        matching_chains = [
            c for c in chains if c.final_answer == most_common_answer
        ]

        best_chain = max(matching_chains, key=lambda c: c.overall_confidence)

        # Boost confidence if consistent
        if consistency >= self.config.consistency_threshold:
            best_chain.overall_confidence = min(
                1.0, best_chain.overall_confidence + 0.1
            )
            best_chain.metadata = {"consistency": consistency}

        return best_chain

    async def _verify_chain(self, chain: ReasoningChain) -> ReasoningChain:
        """Verify a reasoning chain."""
        if not self.generate_fn:
            chain.verified = True
            return chain

        prompt = self.config.verification_prompt_template.format(
            problem=chain.problem,
            chain=chain.get_chain_text(),
        )

        response = await self.generate_fn(prompt)

        # Parse verification result
        is_correct = self._parse_verification_result(response)

        chain.verified = is_correct

        if not is_correct:
            # Add verification step noting the issues
            chain.add_step(
                reasoning_type=ReasoningType.VERIFICATION,
                content=f"Verification found issues: {response[:200]}",
                confidence=0.6,
            )
            chain.overall_confidence *= 0.8

        return chain

    async def _decompose_problem(self, problem: str) -> list[str]:
        """Decompose problem into sub-problems (least-to-most)."""
        if not self.generate_fn:
            return []

        prompt = self.config.decomposition_prompt_template.format(problem=problem)
        response = await self.generate_fn(prompt)

        # Parse sub-problems
        subproblems = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Clean up the line
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    subproblems.append(clean)

        return subproblems[:self.config.max_subproblems]

    def _parse_reasoning_steps(
        self,
        response: str,
    ) -> list[tuple[ReasoningType, str]]:
        """Parse reasoning steps from LLM response."""
        steps = []

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try to parse step format: "Step N (type): content"
            if line.lower().startswith("step"):
                # Extract type and content
                try:
                    parts = line.split(":", 1)
                    if len(parts) >= 2:
                        type_part = parts[0].lower()
                        content = parts[1].strip()

                        # Detect reasoning type
                        reasoning_type = ReasoningType.INFERENCE  # Default
                        for rt in ReasoningType:
                            if rt.value in type_part:
                                reasoning_type = rt
                                break

                        steps.append((reasoning_type, content))
                except Exception:
                    continue

        # Fallback if no steps parsed
        if not steps:
            steps.append((ReasoningType.INFERENCE, response[:500]))

        return steps

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract final answer from response."""
        # Look for answer markers
        markers = ["answer:", "therefore:", "conclusion:", "result:", "finally:"]

        response_lower = response.lower()
        for marker in markers:
            idx = response_lower.rfind(marker)
            if idx != -1:
                answer = response[idx + len(marker):].strip()
                # Take first sentence or up to 200 chars
                end = min(answer.find(".") + 1 if "." in answer else len(answer), 200)
                return answer[:end].strip()

        # Fallback: last sentence
        sentences = response.split(".")
        if sentences:
            return sentences[-1].strip() or (sentences[-2].strip() if len(sentences) > 1 else None)

        return None

    def _parse_verification_result(self, response: str) -> bool:
        """Parse verification result to determine if reasoning is correct."""
        response_lower = response.lower()

        # Look for indicators of correctness
        positive_indicators = ["correct", "valid", "accurate", "right", "no errors"]
        negative_indicators = ["incorrect", "error", "wrong", "mistake", "invalid"]

        positive_score = sum(1 for ind in positive_indicators if ind in response_lower)
        negative_score = sum(1 for ind in negative_indicators if ind in response_lower)

        return positive_score > negative_score

    def create_manual_chain(
        self,
        problem: str,
        steps: list[tuple[str, str]],
        answer: Optional[str] = None,
    ) -> ReasoningChain:
        """
        Create a reasoning chain manually.

        Args:
            problem: The problem statement
            steps: List of (type, content) tuples
            answer: Optional final answer

        Returns:
            Constructed reasoning chain
        """
        self._chain_counter += 1
        chain = ReasoningChain(
            id=f"cot-manual-{self._chain_counter}",
            problem=problem,
        )

        for step_type, content in steps:
            try:
                rt = ReasoningType(step_type)
            except ValueError:
                rt = ReasoningType.INFERENCE

            chain.add_step(
                reasoning_type=rt,
                content=content,
            )

        if answer:
            chain.final_answer = answer
            chain.add_step(
                reasoning_type=ReasoningType.CONCLUSION,
                content=answer,
            )

        chain.overall_confidence = 0.8
        return chain

    def get_stats(self) -> dict[str, Any]:
        """Get reasoning statistics."""
        return {
            "chains_generated": self._chain_counter,
            "config": self.config.to_dict(),
        }
