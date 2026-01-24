"""
Memory Consolidation System

Implements memory consolidation processes inspired by human sleep-based
memory consolidation, including pattern extraction, knowledge distillation,
and memory optimization.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
import json

import structlog

from .episodic import EpisodicMemory, Episode, EpisodeType
from .semantic import SemanticMemory, RelationType, KnowledgeTriple
from .vector_store import VectorStore

logger = structlog.get_logger()


class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation."""

    REPLAY = "replay"  # Experience replay
    ABSTRACTION = "abstraction"  # Extract abstract patterns
    INTERLEAVING = "interleaving"  # Mix similar and different memories
    SPACED = "spaced"  # Spaced repetition based
    SCHEMA = "schema"  # Schema-based consolidation


@dataclass
class ConsolidationResult:
    """Result of a consolidation cycle."""

    strategy: ConsolidationStrategy
    episodes_processed: int
    patterns_extracted: int
    knowledge_added: int
    memories_pruned: int
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "episodes_processed": self.episodes_processed,
            "patterns_extracted": self.patterns_extracted,
            "knowledge_added": self.knowledge_added,
            "memories_pruned": self.memories_pruned,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExtractedPattern:
    """A pattern extracted during consolidation."""

    id: str
    pattern_type: str  # "action_sequence", "causal", "conditional", etc.
    description: str
    conditions: list[str]
    actions: list[str]
    expected_outcome: str
    confidence: float
    support_count: int  # Number of episodes supporting this pattern
    source_episodes: list[str]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "conditions": self.conditions,
            "actions": self.actions,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "source_episodes": self.source_episodes,
            "created_at": self.created_at.isoformat(),
        }


class MemoryConsolidator:
    """
    Memory consolidation system.

    Features:
    - Periodic consolidation cycles
    - Pattern extraction from episodes
    - Knowledge graph enrichment
    - Memory importance updates
    - Forgetting curve implementation
    - Cross-modal memory integration
    """

    def __init__(
        self,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        vector_store: Optional[VectorStore] = None,
        consolidation_interval: timedelta = timedelta(hours=1),
        min_pattern_support: int = 3,
        forgetting_rate: float = 0.05,
    ):
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.vector_store = vector_store
        self.consolidation_interval = consolidation_interval
        self.min_pattern_support = min_pattern_support
        self.forgetting_rate = forgetting_rate

        # Storage
        self._patterns: dict[str, ExtractedPattern] = {}
        self._consolidation_history: list[ConsolidationResult] = []

        # State
        self._last_consolidation: Optional[datetime] = None
        self._consolidation_task: Optional[asyncio.Task] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize consolidator."""
        self._initialized = True
        # Start background consolidation
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        logger.info("memory_consolidator_initialized")

    async def shutdown(self) -> None:
        """Shutdown consolidator."""
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        self._initialized = False
        logger.info("memory_consolidator_shutdown")

    async def consolidate(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.REPLAY,
    ) -> ConsolidationResult:
        """
        Run a consolidation cycle.

        Args:
            strategy: The consolidation strategy to use

        Returns:
            Result of the consolidation cycle
        """
        start_time = datetime.now()
        episodes_processed = 0
        patterns_extracted = 0
        knowledge_added = 0
        memories_pruned = 0

        logger.info("consolidation_started", strategy=strategy.value)

        try:
            if strategy == ConsolidationStrategy.REPLAY:
                result = await self._replay_consolidation()
            elif strategy == ConsolidationStrategy.ABSTRACTION:
                result = await self._abstraction_consolidation()
            elif strategy == ConsolidationStrategy.INTERLEAVING:
                result = await self._interleaving_consolidation()
            elif strategy == ConsolidationStrategy.SPACED:
                result = await self._spaced_consolidation()
            elif strategy == ConsolidationStrategy.SCHEMA:
                result = await self._schema_consolidation()
            else:
                result = {"episodes": 0, "patterns": 0, "knowledge": 0, "pruned": 0}

            episodes_processed = result.get("episodes", 0)
            patterns_extracted = result.get("patterns", 0)
            knowledge_added = result.get("knowledge", 0)
            memories_pruned = result.get("pruned", 0)

        except Exception as e:
            logger.error("consolidation_error", error=str(e))

        duration = (datetime.now() - start_time).total_seconds()
        self._last_consolidation = datetime.now()

        consolidation_result = ConsolidationResult(
            strategy=strategy,
            episodes_processed=episodes_processed,
            patterns_extracted=patterns_extracted,
            knowledge_added=knowledge_added,
            memories_pruned=memories_pruned,
            duration_seconds=duration,
        )

        self._consolidation_history.append(consolidation_result)

        # Keep history bounded
        if len(self._consolidation_history) > 100:
            self._consolidation_history = self._consolidation_history[-100:]

        logger.info(
            "consolidation_completed",
            strategy=strategy.value,
            episodes=episodes_processed,
            patterns=patterns_extracted,
            duration=duration,
        )

        return consolidation_result

    async def _replay_consolidation(self) -> dict[str, int]:
        """
        Experience replay consolidation.

        Samples episodes for replay and strengthens important memories.
        """
        if not self.episodic_memory:
            return {"episodes": 0, "patterns": 0, "knowledge": 0, "pruned": 0}

        # Sample episodes for replay
        episodes = await self.episodic_memory.sample_for_replay(
            n=20,
            prioritized=True,
        )

        patterns_found = 0
        knowledge_added = 0

        # Process each episode
        for episode in episodes:
            # Extract patterns from successful episodes
            if episode.success:
                patterns = await self._extract_patterns(episode)
                patterns_found += len(patterns)

                # Add to semantic memory
                if self.semantic_memory and episode.lessons_learned:
                    for lesson in episode.lessons_learned:
                        triple = KnowledgeTriple(
                            subject=episode.title,
                            predicate="teaches",
                            object=lesson,
                            confidence=0.8,
                            source="consolidation",
                        )
                        await self.semantic_memory.add_triple(triple)
                        knowledge_added += 1

            # Update importance based on replay
            await self.episodic_memory.update_importance(
                episode.id,
                importance_delta=0.1 if episode.success else 0.05,
            )

        return {
            "episodes": len(episodes),
            "patterns": patterns_found,
            "knowledge": knowledge_added,
            "pruned": 0,
        }

    async def _abstraction_consolidation(self) -> dict[str, int]:
        """
        Abstraction-based consolidation.

        Extracts abstract patterns from concrete episodes.
        """
        if not self.episodic_memory:
            return {"episodes": 0, "patterns": 0, "knowledge": 0, "pruned": 0}

        # Get recent episodes by type
        episodes_by_type: dict[EpisodeType, list[Episode]] = {}

        for ep_type in EpisodeType:
            episodes = await self.episodic_memory.search_temporal(
                start_time=datetime.now() - timedelta(days=7),
                limit=50,
            )
            type_episodes = [e for e in episodes if e.episode_type == ep_type]
            if type_episodes:
                episodes_by_type[ep_type] = type_episodes

        patterns_found = 0
        knowledge_added = 0

        for ep_type, episodes in episodes_by_type.items():
            if len(episodes) >= self.min_pattern_support:
                # Find common action sequences
                action_sequences = self._find_common_sequences(episodes)

                for sequence, support in action_sequences.items():
                    if support >= self.min_pattern_support:
                        pattern_id = f"pat-{ep_type.value}-{len(self._patterns)}"

                        pattern = ExtractedPattern(
                            id=pattern_id,
                            pattern_type="action_sequence",
                            description=f"Common sequence for {ep_type.value}",
                            conditions=[],
                            actions=list(sequence),
                            expected_outcome="Task completion",
                            confidence=support / len(episodes),
                            support_count=support,
                            source_episodes=[e.id for e in episodes[:support]],
                        )

                        self._patterns[pattern_id] = pattern
                        patterns_found += 1

                        # Add pattern to semantic memory
                        if self.semantic_memory:
                            await self.semantic_memory.add_concept(
                                name=f"Pattern: {sequence[0]}",
                                description=pattern.description,
                                concept_type="pattern",
                            )
                            knowledge_added += 1

        return {
            "episodes": sum(len(eps) for eps in episodes_by_type.values()),
            "patterns": patterns_found,
            "knowledge": knowledge_added,
            "pruned": 0,
        }

    async def _interleaving_consolidation(self) -> dict[str, int]:
        """
        Interleaved consolidation.

        Mixes similar and different memories to improve generalization.
        """
        if not self.episodic_memory:
            return {"episodes": 0, "patterns": 0, "knowledge": 0, "pruned": 0}

        # Get success and failure episodes
        success_episodes = await self.episodic_memory.search_similar(
            query="success completion",
            k=10,
            success_only=True,
        )

        failure_episodes = await self.episodic_memory.search_similar(
            query="error failure",
            k=10,
            success_only=False,
        )
        failure_episodes = [e for e in failure_episodes if not e.success]

        knowledge_added = 0

        # Interleave processing
        all_episodes = []
        for i in range(max(len(success_episodes), len(failure_episodes))):
            if i < len(success_episodes):
                all_episodes.append(success_episodes[i])
            if i < len(failure_episodes):
                all_episodes.append(failure_episodes[i])

        # Extract contrasting patterns
        for i, ep1 in enumerate(all_episodes[:-1]):
            ep2 = all_episodes[i + 1]

            if ep1.success != ep2.success:
                # Contrastive pair found
                if self.semantic_memory:
                    # Add causal knowledge
                    if ep1.success:
                        triple = KnowledgeTriple(
                            subject=ep1.outcome,
                            predicate="contrasts_with",
                            object=ep2.outcome,
                            confidence=0.7,
                        )
                    else:
                        triple = KnowledgeTriple(
                            subject=ep2.outcome,
                            predicate="contrasts_with",
                            object=ep1.outcome,
                            confidence=0.7,
                        )

                    await self.semantic_memory.add_triple(triple)
                    knowledge_added += 1

        return {
            "episodes": len(all_episodes),
            "patterns": 0,
            "knowledge": knowledge_added,
            "pruned": 0,
        }

    async def _spaced_consolidation(self) -> dict[str, int]:
        """
        Spaced repetition consolidation.

        Updates memory importance based on spaced repetition schedule.
        """
        if not self.episodic_memory:
            return {"episodes": 0, "patterns": 0, "knowledge": 0, "pruned": 0}

        episodes_updated = 0
        memories_pruned = 0

        # Get all episodes with access history
        all_episodes = await self.episodic_memory.search_temporal(limit=1000)

        for episode in all_episodes:
            # Calculate optimal review time based on access count
            optimal_interval = timedelta(days=2 ** episode.access_count)

            if episode.last_accessed:
                time_since_access = datetime.now() - episode.last_accessed

                if time_since_access > optimal_interval:
                    # Due for review - boost importance
                    await self.episodic_memory.update_importance(
                        episode.id,
                        importance_delta=0.1,
                    )
                    episodes_updated += 1

                elif time_since_access > optimal_interval * 3:
                    # Significantly overdue - apply forgetting
                    new_importance = episode.importance * (1 - self.forgetting_rate)
                    await self.episodic_memory.update_importance(
                        episode.id,
                        importance_delta=new_importance - episode.importance,
                    )

                    # Prune if importance too low
                    if new_importance < 0.1:
                        memories_pruned += 1

        return {
            "episodes": len(all_episodes),
            "patterns": 0,
            "knowledge": 0,
            "pruned": memories_pruned,
        }

    async def _schema_consolidation(self) -> dict[str, int]:
        """
        Schema-based consolidation.

        Organizes memories according to schemas/frames.
        """
        if not self.episodic_memory or not self.semantic_memory:
            return {"episodes": 0, "patterns": 0, "knowledge": 0, "pruned": 0}

        knowledge_added = 0

        # Define basic schemas
        schemas = {
            "problem_solving": {
                "slots": ["problem", "approach", "solution", "outcome"],
                "types": [EpisodeType.TASK_EXECUTION, EpisodeType.PLANNING],
            },
            "learning": {
                "slots": ["topic", "method", "insight", "application"],
                "types": [EpisodeType.LEARNING],
            },
            "collaboration": {
                "slots": ["participants", "goal", "process", "result"],
                "types": [EpisodeType.COLLABORATION],
            },
        }

        for schema_name, schema_def in schemas.items():
            # Get relevant episodes
            episodes = []
            for ep_type in schema_def["types"]:
                type_episodes = await self.episodic_memory.search_temporal(
                    start_time=datetime.now() - timedelta(days=30),
                    limit=20,
                )
                episodes.extend([e for e in type_episodes if e.episode_type == ep_type])

            # Create schema concept
            schema_concept = await self.semantic_memory.add_concept(
                name=schema_name,
                description=f"Schema for {schema_name}",
                concept_type="schema",
                properties={"slots": schema_def["slots"]},
            )

            # Link episodes to schema
            for episode in episodes:
                await self.semantic_memory.add_relation(
                    source=episode.title,
                    relation_type=RelationType.INSTANCE_OF,
                    target=schema_concept,
                    confidence=0.7,
                )
                knowledge_added += 1

        return {
            "episodes": sum(
                len([e for e in await self.episodic_memory.search_temporal(limit=100) if e.episode_type in schema_def["types"]])
                for schema_def in schemas.values()
            ),
            "patterns": len(schemas),
            "knowledge": knowledge_added,
            "pruned": 0,
        }

    async def _extract_patterns(self, episode: Episode) -> list[ExtractedPattern]:
        """Extract patterns from a single episode."""
        patterns = []

        # Extract action sequence pattern
        if len(episode.steps) >= 2:
            actions = [step.action for step in episode.steps]
            action_tuple = tuple(actions[:5])  # Limit to first 5 actions

            pattern_id = f"pat-{episode.id}-seq"
            pattern = ExtractedPattern(
                id=pattern_id,
                pattern_type="action_sequence",
                description=f"Sequence from {episode.title}",
                conditions=[],
                actions=list(action_tuple),
                expected_outcome=episode.outcome,
                confidence=0.8 if episode.success else 0.3,
                support_count=1,
                source_episodes=[episode.id],
            )

            self._patterns[pattern_id] = pattern
            patterns.append(pattern)

        # Extract conditional pattern
        if episode.context and episode.steps:
            pattern_id = f"pat-{episode.id}-cond"
            pattern = ExtractedPattern(
                id=pattern_id,
                pattern_type="conditional",
                description=f"Conditional pattern from {episode.title}",
                conditions=list(episode.context.keys())[:3],
                actions=[episode.steps[0].action],
                expected_outcome=episode.outcome,
                confidence=0.7,
                support_count=1,
                source_episodes=[episode.id],
            )

            self._patterns[pattern_id] = pattern
            patterns.append(pattern)

        return patterns

    def _find_common_sequences(
        self,
        episodes: list[Episode],
    ) -> dict[tuple[str, ...], int]:
        """Find common action sequences across episodes."""
        sequence_counts: dict[tuple[str, ...], int] = {}

        for episode in episodes:
            if len(episode.steps) < 2:
                continue

            actions = tuple(step.action for step in episode.steps[:5])

            # Count full sequence
            if actions in sequence_counts:
                sequence_counts[actions] += 1
            else:
                sequence_counts[actions] = 1

            # Count subsequences
            for i in range(len(actions)):
                for j in range(i + 2, len(actions) + 1):
                    subseq = actions[i:j]
                    if subseq in sequence_counts:
                        sequence_counts[subseq] += 1
                    else:
                        sequence_counts[subseq] = 1

        return sequence_counts

    async def _consolidation_loop(self) -> None:
        """Background consolidation loop."""
        while True:
            try:
                await asyncio.sleep(self.consolidation_interval.total_seconds())

                # Run consolidation with different strategies
                strategies = [
                    ConsolidationStrategy.REPLAY,
                    ConsolidationStrategy.ABSTRACTION,
                    ConsolidationStrategy.SPACED,
                ]

                for strategy in strategies:
                    await self.consolidate(strategy)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("consolidation_loop_error", error=str(e))

    def get_patterns(self, pattern_type: Optional[str] = None) -> list[ExtractedPattern]:
        """Get extracted patterns."""
        patterns = list(self._patterns.values())
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        return patterns

    def get_stats(self) -> dict[str, Any]:
        """Get consolidator statistics."""
        return {
            "patterns_extracted": len(self._patterns),
            "consolidations_run": len(self._consolidation_history),
            "last_consolidation": self._last_consolidation.isoformat() if self._last_consolidation else None,
            "recent_results": [r.to_dict() for r in self._consolidation_history[-5:]],
        }
