"""
LLM Integration for SOTA Agent Features

Provides real LLM integration for:
- Tree-of-Thought reasoning
- Chain-of-Thought reasoning
- Embedding generation for vector memory
- Self-reflection and metacognition
- Learning feedback evaluation

Designed to work with Llama 3.3 70B via OpenAI-compatible API (vLLM/Ollama).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

import structlog

from aion.core.llm import (
    LLMAdapter,
    LLMConfig,
    LLMProvider,
    Message,
    LLMResponse,
    create_llm_adapter,
)

logger = structlog.get_logger(__name__)


class ModelType(str, Enum):
    """Types of models used in SOTA features."""

    REASONING = "reasoning"  # Main LLM for reasoning (Llama 3.3 70B)
    EMBEDDING = "embedding"  # Embedding model
    EVALUATION = "evaluation"  # For evaluating outputs


@dataclass
class LLMIntegrationConfig:
    """Configuration for LLM integration."""

    # Main reasoning model (Llama 3.3 70B)
    reasoning_provider: LLMProvider = LLMProvider.LOCAL
    reasoning_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    reasoning_base_url: str = "http://localhost:8000/v1"
    reasoning_api_key: Optional[str] = None

    # Embedding model
    embedding_provider: LLMProvider = LLMProvider.LOCAL
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_base_url: str = "http://localhost:8001/v1"
    embedding_dimension: int = 1024

    # Generation settings
    max_tokens: int = 4096
    temperature: float = 0.7
    reasoning_temperature: float = 0.3  # Lower for more focused reasoning

    # Timeouts
    timeout: float = 120.0  # Longer for large models
    embedding_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "LLMIntegrationConfig":
        """Create config from environment variables."""
        return cls(
            reasoning_provider=LLMProvider(
                os.environ.get("AION_LLM_PROVIDER", "local")
            ),
            reasoning_model=os.environ.get(
                "AION_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"
            ),
            reasoning_base_url=os.environ.get(
                "AION_LLM_BASE_URL", "http://localhost:8000/v1"
            ),
            reasoning_api_key=os.environ.get("AION_LLM_API_KEY"),
            embedding_model=os.environ.get(
                "AION_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
            ),
            embedding_base_url=os.environ.get(
                "AION_EMBEDDING_BASE_URL", "http://localhost:8001/v1"
            ),
            embedding_dimension=int(
                os.environ.get("AION_EMBEDDING_DIMENSION", "1024")
            ),
        )


class SOTALLMProvider:
    """
    Unified LLM provider for SOTA agent features.

    Provides high-level methods for:
    - Generating thoughts for ToT/CoT
    - Evaluating reasoning paths
    - Generating embeddings
    - Self-reflection and critique
    """

    _instance: Optional["SOTALLMProvider"] = None

    def __init__(self, config: Optional[LLMIntegrationConfig] = None):
        self.config = config or LLMIntegrationConfig.from_env()

        self._reasoning_adapter: Optional[LLMAdapter] = None
        self._embedding_client: Optional[Any] = None

        self._initialized = False
        self._call_stats: dict[str, int] = {
            "thought_generations": 0,
            "evaluations": 0,
            "embeddings": 0,
            "reflections": 0,
        }

    @classmethod
    async def get_instance(cls) -> "SOTALLMProvider":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()
        return cls._instance

    async def initialize(self) -> None:
        """Initialize LLM adapters."""
        if self._initialized:
            return

        # Initialize reasoning model
        reasoning_config = LLMConfig(
            provider=self.config.reasoning_provider,
            model=self.config.reasoning_model,
            api_key=self.config.reasoning_api_key or "not-needed-for-local",
            base_url=self.config.reasoning_base_url,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )

        self._reasoning_adapter = LLMAdapter(reasoning_config)
        await self._reasoning_adapter.initialize()

        self._initialized = True
        logger.info(
            "sota_llm_provider_initialized",
            reasoning_model=self.config.reasoning_model,
            embedding_model=self.config.embedding_model,
        )

    async def shutdown(self) -> None:
        """Shutdown adapters."""
        if self._reasoning_adapter:
            await self._reasoning_adapter.close()
        self._initialized = False
        logger.info("sota_llm_provider_shutdown")

    # ==========================================
    # Thought Generation (for ToT/CoT)
    # ==========================================

    async def generate_thoughts(
        self,
        problem: str,
        current_path: list[str],
        n: int = 3,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """
        Generate n next thought steps for Tree-of-Thought.

        Args:
            problem: The problem being solved
            current_path: Current reasoning path
            n: Number of thoughts to generate
            temperature: Optional temperature override

        Returns:
            List of generated thoughts
        """
        if not self._initialized:
            await self.initialize()

        path_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(current_path))

        prompt = f"""You are solving a problem step by step. Given the current reasoning path, generate the next logical thought step.

Problem: {problem}

Current reasoning path:
{path_str if current_path else "No steps yet - start reasoning from scratch."}

Generate ONE clear, logical next step in the reasoning process. Be specific and actionable.
Focus on moving closer to the solution.

Next thought step:"""

        thoughts = []
        temp = temperature or self.config.reasoning_temperature

        for i in range(n):
            try:
                # Vary temperature slightly for diversity
                varied_temp = temp + (i * 0.1)

                response = await self._reasoning_adapter.complete(
                    messages=[
                        Message(role="system", content="You are a careful, logical reasoner. Generate one clear reasoning step."),
                        Message(role="user", content=prompt),
                    ],
                    temperature=min(1.0, varied_temp),
                    max_tokens=512,
                )

                thought = response.content.strip()
                if thought and thought not in thoughts:
                    thoughts.append(thought)

            except Exception as e:
                logger.warning("thought_generation_error", error=str(e), attempt=i)

        self._call_stats["thought_generations"] += 1
        return thoughts

    async def generate_cot_reasoning(
        self,
        problem: str,
        context: Optional[str] = None,
    ) -> list[str]:
        """
        Generate a complete Chain-of-Thought reasoning chain.

        Args:
            problem: The problem to solve
            context: Optional additional context

        Returns:
            List of reasoning steps
        """
        if not self._initialized:
            await self.initialize()

        context_str = f"\nContext: {context}" if context else ""

        prompt = f"""Solve this problem step by step. Show your complete reasoning process.

Problem: {problem}{context_str}

Think through this carefully, breaking it down into clear logical steps.
Format each step on a new line starting with "Step N:".

Reasoning:"""

        try:
            response = await self._reasoning_adapter.complete(
                messages=[
                    Message(
                        role="system",
                        content="You are a careful reasoner. Break down problems into clear, logical steps. Always show your work."
                    ),
                    Message(role="user", content=prompt),
                ],
                temperature=self.config.reasoning_temperature,
                max_tokens=2048,
            )

            # Parse steps from response
            content = response.content.strip()
            steps = []

            # Try to extract numbered steps
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    # Remove step numbering if present
                    cleaned = re.sub(r'^(Step\s*\d+[:.]\s*|\d+[.)]\s*)', '', line)
                    if cleaned:
                        steps.append(cleaned)

            self._call_stats["thought_generations"] += 1
            return steps if steps else [content]

        except Exception as e:
            logger.error("cot_generation_error", error=str(e))
            return [f"Error generating reasoning: {str(e)}"]

    # ==========================================
    # Path Evaluation
    # ==========================================

    async def evaluate_reasoning_path(
        self,
        problem: str,
        path: list[str],
    ) -> float:
        """
        Evaluate how promising a reasoning path is.

        Args:
            problem: The original problem
            path: The reasoning path to evaluate

        Returns:
            Score from 0.0 to 1.0
        """
        if not self._initialized:
            await self.initialize()

        path_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(path))

        prompt = f"""Evaluate how promising this reasoning path is for solving the problem.

Problem: {problem}

Reasoning path:
{path_str}

Consider:
1. Is the reasoning logically sound?
2. Does it make progress toward a solution?
3. Are there any errors or gaps?
4. How close is it to a complete solution?

Rate this path from 0.0 (completely wrong/unhelpful) to 1.0 (correct and complete solution).

Respond with ONLY a single decimal number between 0.0 and 1.0:"""

        try:
            response = await self._reasoning_adapter.complete(
                messages=[
                    Message(
                        role="system",
                        content="You are an evaluator. Respond only with a decimal number between 0.0 and 1.0."
                    ),
                    Message(role="user", content=prompt),
                ],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=16,
            )

            # Parse score from response
            content = response.content.strip()

            # Extract first number found
            match = re.search(r'([0-9]*\.?[0-9]+)', content)
            if match:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5  # Default if parsing fails

            self._call_stats["evaluations"] += 1
            return score

        except Exception as e:
            logger.warning("evaluation_error", error=str(e))
            return 0.5

    # ==========================================
    # Self-Reflection
    # ==========================================

    async def reflect_on_output(
        self,
        task: str,
        output: str,
        goal: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate self-reflection on an output.

        Args:
            task: The original task
            output: The generated output
            goal: Optional goal description

        Returns:
            Reflection with critique and suggestions
        """
        if not self._initialized:
            await self.initialize()

        goal_str = f"\nGoal: {goal}" if goal else ""

        prompt = f"""Reflect on this output and provide constructive critique.

Task: {task}{goal_str}

Output:
{output}

Analyze the output considering:
1. Does it fully address the task?
2. Are there any errors or inaccuracies?
3. What could be improved?
4. What was done well?

Provide your reflection in JSON format:
{{
    "overall_quality": <0.0-1.0>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "is_complete": <true/false>
}}

JSON:"""

        try:
            response = await self._reasoning_adapter.complete(
                messages=[
                    Message(
                        role="system",
                        content="You are a thoughtful critic. Provide balanced, constructive feedback in valid JSON format."
                    ),
                    Message(role="user", content=prompt),
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                reflection = json.loads(json_match.group())
            else:
                reflection = {
                    "overall_quality": 0.5,
                    "strengths": [],
                    "weaknesses": ["Could not parse reflection"],
                    "suggestions": [],
                    "is_complete": False,
                    "raw_response": content,
                }

            self._call_stats["reflections"] += 1
            return reflection

        except Exception as e:
            logger.warning("reflection_error", error=str(e))
            return {
                "overall_quality": 0.5,
                "strengths": [],
                "weaknesses": [f"Reflection error: {str(e)}"],
                "suggestions": [],
                "is_complete": False,
            }

    async def critique_and_improve(
        self,
        task: str,
        output: str,
        max_iterations: int = 2,
    ) -> tuple[str, list[dict]]:
        """
        Iteratively critique and improve an output.

        Args:
            task: The original task
            output: The initial output
            max_iterations: Maximum improvement iterations

        Returns:
            Tuple of (improved_output, iteration_history)
        """
        current_output = output
        history = []

        for i in range(max_iterations):
            # Get reflection
            reflection = await self.reflect_on_output(task, current_output)
            history.append({
                "iteration": i + 1,
                "reflection": reflection,
                "output": current_output,
            })

            # Check if good enough
            if reflection.get("overall_quality", 0) >= 0.9:
                break

            # Generate improved version
            suggestions = reflection.get("suggestions", [])
            if not suggestions:
                break

            prompt = f"""Improve this output based on the feedback.

Task: {task}

Current output:
{current_output}

Feedback to address:
{chr(10).join(f"- {s}" for s in suggestions)}

Provide an improved version:"""

            try:
                response = await self._reasoning_adapter.complete(
                    messages=[
                        Message(role="system", content="Improve the output based on feedback."),
                        Message(role="user", content=prompt),
                    ],
                    temperature=0.5,
                    max_tokens=2048,
                )

                current_output = response.content.strip()

            except Exception as e:
                logger.warning("improvement_error", error=str(e))
                break

        return current_output, history

    # ==========================================
    # Embeddings
    # ==========================================

    async def generate_embedding(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self._initialized:
            await self.initialize()

        # Try to use embedding endpoint if available
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.config.embedding_timeout) as client:
                response = await client.post(
                    f"{self.config.embedding_base_url}/embeddings",
                    json={
                        "model": self.config.embedding_model,
                        "input": text,
                    },
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data["data"][0]["embedding"]
                    self._call_stats["embeddings"] += 1
                    return embedding

        except Exception as e:
            logger.debug("embedding_api_fallback", error=str(e))

        # Fallback: Use LLM to generate semantic hash
        # This is a workaround when no embedding model is available
        return await self._llm_based_embedding(text)

    async def _llm_based_embedding(self, text: str) -> list[float]:
        """
        Generate a pseudo-embedding using LLM semantic features.

        This is a fallback when no embedding model is available.
        Uses LLM to extract semantic features and creates a consistent vector.
        """
        prompt = f"""Extract the key semantic features from this text.
List exactly 10 key concepts/themes, one per line:

Text: {text[:500]}

Key concepts:"""

        try:
            response = await self._reasoning_adapter.complete(
                messages=[
                    Message(role="system", content="Extract key semantic features."),
                    Message(role="user", content=prompt),
                ],
                temperature=0.1,
                max_tokens=256,
            )

            # Create embedding from semantic features
            features = response.content.strip().split('\n')[:10]

            # Generate consistent vector from features
            dimension = self.config.embedding_dimension
            vector = [0.0] * dimension

            for i, feature in enumerate(features):
                feature_hash = hashlib.md5(feature.lower().encode()).hexdigest()
                for j, char in enumerate(feature_hash):
                    idx = (i * 32 + j) % dimension
                    vector[idx] += ord(char) / 255.0

            # Add original text hash for uniqueness
            text_hash = hashlib.md5(text.encode()).hexdigest()
            for i, char in enumerate(text_hash):
                idx = i % dimension
                vector[idx] += ord(char) / 1000.0

            # Normalize
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]

            self._call_stats["embeddings"] += 1
            return vector

        except Exception as e:
            logger.warning("llm_embedding_error", error=str(e))
            # Last resort: simple hash embedding
            return self._simple_hash_embedding(text)

    def _simple_hash_embedding(self, text: str) -> list[float]:
        """Simple hash-based embedding as last resort."""
        dimension = self.config.embedding_dimension
        vector = [0.0] * dimension

        for i, char in enumerate(text):
            idx = (ord(char) + i) % dimension
            vector[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    async def generate_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        tasks = [self.generate_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)

    # ==========================================
    # Learning Feedback
    # ==========================================

    async def evaluate_action_outcome(
        self,
        state: dict[str, Any],
        action: str,
        outcome: dict[str, Any],
    ) -> float:
        """
        Evaluate an action's outcome for RL feedback.

        Args:
            state: State before action
            action: Action taken
            outcome: Result of action

        Returns:
            Reward signal from -1.0 to 1.0
        """
        if not self._initialized:
            await self.initialize()

        prompt = f"""Evaluate the outcome of this action.

Initial state: {json.dumps(state, default=str)[:500]}
Action taken: {action}
Outcome: {json.dumps(outcome, default=str)[:500]}

Was this action successful? Consider:
1. Did it achieve its intended purpose?
2. Were there any negative side effects?
3. Was it efficient?

Rate the outcome from -1.0 (very bad) to 1.0 (excellent).
Respond with only a number:"""

        try:
            response = await self._reasoning_adapter.complete(
                messages=[
                    Message(role="system", content="Rate action outcomes. Respond only with a number."),
                    Message(role="user", content=prompt),
                ],
                temperature=0.1,
                max_tokens=16,
            )

            match = re.search(r'(-?[0-9]*\.?[0-9]+)', response.content.strip())
            if match:
                reward = float(match.group(1))
                return max(-1.0, min(1.0, reward))

            return 0.0

        except Exception as e:
            logger.warning("reward_evaluation_error", error=str(e))
            return 0.0

    # ==========================================
    # Utility Methods
    # ==========================================

    async def extract_answer(
        self,
        reasoning: list[str],
        problem: str,
    ) -> str:
        """
        Extract final answer from reasoning chain.

        Args:
            reasoning: The reasoning steps
            problem: Original problem

        Returns:
            Extracted answer
        """
        if not self._initialized:
            await self.initialize()

        reasoning_str = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reasoning))

        prompt = f"""Based on this reasoning, provide a clear, concise final answer.

Problem: {problem}

Reasoning:
{reasoning_str}

Final answer:"""

        try:
            response = await self._reasoning_adapter.complete(
                messages=[
                    Message(role="system", content="Provide a clear, concise answer."),
                    Message(role="user", content=prompt),
                ],
                temperature=0.2,
                max_tokens=512,
            )

            return response.content.strip()

        except Exception as e:
            logger.warning("answer_extraction_error", error=str(e))
            return reasoning[-1] if reasoning else "Unable to extract answer"

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "initialized": self._initialized,
            "reasoning_model": self.config.reasoning_model,
            "embedding_model": self.config.embedding_model,
            **self._call_stats,
        }


# ==========================================
# Callback Functions for SOTA Modules
# ==========================================

async def create_thought_generator() -> Callable[[str, list[str], int], Awaitable[list[str]]]:
    """Create a thought generator callback for ToT."""
    provider = await SOTALLMProvider.get_instance()

    async def generate(problem: str, path: list[str], n: int) -> list[str]:
        return await provider.generate_thoughts(problem, path, n)

    return generate


async def create_path_evaluator() -> Callable[[str, list[str]], Awaitable[float]]:
    """Create a path evaluator callback for ToT."""
    provider = await SOTALLMProvider.get_instance()

    async def evaluate(problem: str, path: list[str]) -> float:
        return await provider.evaluate_reasoning_path(problem, path)

    return evaluate


async def create_embedding_function() -> Callable[[str], Awaitable[list[float]]]:
    """Create an embedding function for vector store."""
    provider = await SOTALLMProvider.get_instance()

    async def embed(text: str) -> list[float]:
        return await provider.generate_embedding(text)

    return embed
