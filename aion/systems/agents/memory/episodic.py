"""
Episodic Memory System

Stores and retrieves experiences as episodes, enabling experience replay,
temporal reasoning, and learning from past interactions.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import heapq

import structlog

from .vector_store import VectorStore, SimilarityMetric

logger = structlog.get_logger()


class EpisodeType(Enum):
    """Types of episodes."""

    TASK_EXECUTION = "task_execution"
    CONVERSATION = "conversation"
    LEARNING = "learning"
    ERROR = "error"
    SUCCESS = "success"
    COLLABORATION = "collaboration"
    PLANNING = "planning"
    REFLECTION = "reflection"


@dataclass
class EpisodeStep:
    """A single step within an episode."""

    step_number: int
    action: str
    observation: str
    thought: str = ""
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "action": self.action,
            "observation": self.observation,
            "thought": self.thought,
            "reward": self.reward,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeStep":
        """Create from dictionary."""
        return cls(
            step_number=data["step_number"],
            action=data["action"],
            observation=data["observation"],
            thought=data.get("thought", ""),
            reward=data.get("reward", 0.0),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
        )


@dataclass
class Episode:
    """
    A complete episode representing an experience.

    Episodes contain a sequence of steps representing actions,
    observations, and thoughts during task execution.
    """

    id: str = field(default_factory=lambda: f"ep-{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    episode_type: EpisodeType = EpisodeType.TASK_EXECUTION
    agent_id: str = ""
    title: str = ""
    description: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    steps: list[EpisodeStep] = field(default_factory=list)
    outcome: str = ""
    success: bool = False
    total_reward: float = 0.0
    lessons_learned: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    importance: float = 0.5  # 0-1, for prioritized replay
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def add_step(
        self,
        action: str,
        observation: str,
        thought: str = "",
        reward: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EpisodeStep:
        """Add a step to the episode."""
        step = EpisodeStep(
            step_number=len(self.steps) + 1,
            action=action,
            observation=observation,
            thought=thought,
            reward=reward,
            metadata=metadata or {},
        )
        self.steps.append(step)
        self.total_reward += reward
        return step

    def complete(self, outcome: str, success: bool, lessons: Optional[list[str]] = None) -> None:
        """Mark the episode as complete."""
        self.outcome = outcome
        self.success = success
        self.end_time = datetime.now()
        if lessons:
            self.lessons_learned.extend(lessons)

        # Update importance based on outcome
        if success:
            self.importance = min(1.0, self.importance + 0.2)
        else:
            self.importance = min(1.0, self.importance + 0.3)  # Failures are important to learn from

    @property
    def duration(self) -> Optional[timedelta]:
        """Get episode duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def get_summary(self) -> str:
        """Get a text summary of the episode."""
        steps_summary = "\n".join(
            f"  {s.step_number}. {s.action} -> {s.observation[:100]}..."
            for s in self.steps[:5]
        )
        if len(self.steps) > 5:
            steps_summary += f"\n  ... and {len(self.steps) - 5} more steps"

        return f"""Episode: {self.title}
Type: {self.episode_type.value}
Outcome: {self.outcome}
Success: {self.success}
Total Reward: {self.total_reward:.2f}
Steps:
{steps_summary}
Lessons: {', '.join(self.lessons_learned[:3])}"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "episode_type": self.episode_type.value,
            "agent_id": self.agent_id,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "steps": [s.to_dict() for s in self.steps],
            "outcome": self.outcome,
            "success": self.success,
            "total_reward": self.total_reward,
            "lessons_learned": self.lessons_learned,
            "tags": self.tags,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        """Create from dictionary."""
        episode = cls(
            id=data["id"],
            episode_type=EpisodeType(data["episode_type"]),
            agent_id=data.get("agent_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            context=data.get("context", {}),
            steps=[EpisodeStep.from_dict(s) for s in data.get("steps", [])],
            outcome=data.get("outcome", ""),
            success=data.get("success", False),
            total_reward=data.get("total_reward", 0.0),
            lessons_learned=data.get("lessons_learned", []),
            tags=data.get("tags", []),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else datetime.now(),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )
        return episode


class EpisodicMemory:
    """
    Episodic memory system for storing and retrieving experiences.

    Features:
    - Experience storage and retrieval
    - Temporal indexing
    - Similarity-based search via vector embeddings
    - Prioritized experience replay
    - Memory consolidation and forgetting
    - Cross-episode pattern recognition
    """

    def __init__(
        self,
        max_episodes: int = 10000,
        embedding_dimension: int = 1536,
        storage_path: Optional[Path] = None,
        decay_rate: float = 0.01,  # Memory decay rate
        consolidation_threshold: int = 100,  # Episodes before consolidation
    ):
        self.max_episodes = max_episodes
        self.embedding_dimension = embedding_dimension
        self.storage_path = storage_path
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold

        # Storage
        self._episodes: dict[str, Episode] = {}
        self._temporal_index: list[tuple[datetime, str]] = []  # (timestamp, episode_id)
        self._agent_index: dict[str, list[str]] = {}  # agent_id -> episode_ids
        self._tag_index: dict[str, set[str]] = {}  # tag -> episode_ids
        self._type_index: dict[EpisodeType, list[str]] = {t: [] for t in EpisodeType}

        # Vector store for semantic search
        self._vector_store: Optional[VectorStore] = None

        # Statistics
        self._total_episodes_stored = 0
        self._total_episodes_replayed = 0

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize episodic memory."""
        if self._initialized:
            return

        # Initialize vector store
        vector_path = self.storage_path / "episode_vectors" if self.storage_path else None
        self._vector_store = VectorStore(
            dimension=self.embedding_dimension,
            metric=SimilarityMetric.COSINE,
            max_elements=self.max_episodes,
            storage_path=vector_path,
        )
        await self._vector_store.initialize()

        # Load from disk
        if self.storage_path and self.storage_path.exists():
            await self._load_from_disk()

        self._initialized = True
        logger.info("episodic_memory_initialized", max_episodes=self.max_episodes)

    async def shutdown(self) -> None:
        """Shutdown and persist."""
        if self.storage_path:
            await self._save_to_disk()

        if self._vector_store:
            await self._vector_store.shutdown()

        self._initialized = False
        logger.info("episodic_memory_shutdown")

    async def store(self, episode: Episode) -> str:
        """Store an episode in memory."""
        async with self._lock:
            # Check capacity
            if len(self._episodes) >= self.max_episodes:
                await self._forget_least_important()

            # Store episode
            self._episodes[episode.id] = episode
            self._total_episodes_stored += 1

            # Update indexes
            self._temporal_index.append((episode.start_time, episode.id))
            self._temporal_index.sort(key=lambda x: x[0])

            if episode.agent_id:
                if episode.agent_id not in self._agent_index:
                    self._agent_index[episode.agent_id] = []
                self._agent_index[episode.agent_id].append(episode.id)

            for tag in episode.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(episode.id)

            self._type_index[episode.episode_type].append(episode.id)

            # Add to vector store for semantic search
            if self._vector_store:
                episode_text = f"{episode.title} {episode.description} {episode.outcome}"
                await self._vector_store.add(
                    text=episode_text,
                    metadata={
                        "episode_id": episode.id,
                        "type": episode.episode_type.value,
                        "success": episode.success,
                    },
                    entry_id=episode.id,
                )

            logger.debug("episode_stored", episode_id=episode.id, type=episode.episode_type.value)

            # Trigger consolidation if needed
            if len(self._episodes) % self.consolidation_threshold == 0:
                asyncio.create_task(self._consolidate())

            return episode.id

    async def retrieve(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        episode = self._episodes.get(episode_id)
        if episode:
            episode.access_count += 1
            episode.last_accessed = datetime.now()
            self._total_episodes_replayed += 1
        return episode

    async def search_similar(
        self,
        query: str,
        k: int = 5,
        episode_type: Optional[EpisodeType] = None,
        success_only: bool = False,
    ) -> list[Episode]:
        """Search for similar episodes using semantic similarity."""
        if not self._vector_store:
            return []

        # Build filter
        metadata_filter = {}
        if episode_type:
            metadata_filter["type"] = episode_type.value
        if success_only:
            metadata_filter["success"] = True

        results = await self._vector_store.search(
            query=query,
            k=k,
            metadata_filter=metadata_filter if metadata_filter else None,
        )

        episodes = []
        for result in results:
            episode = self._episodes.get(result.entry.id)
            if episode:
                episode.access_count += 1
                episode.last_accessed = datetime.now()
                episodes.append(episode)

        self._total_episodes_replayed += len(episodes)
        return episodes

    async def search_temporal(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Search episodes by time range."""
        results = []

        for timestamp, episode_id in self._temporal_index:
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue

            episode = self._episodes.get(episode_id)
            if episode:
                results.append(episode)
                if len(results) >= limit:
                    break

        return results

    async def search_by_agent(self, agent_id: str, limit: int = 10) -> list[Episode]:
        """Get episodes for a specific agent."""
        episode_ids = self._agent_index.get(agent_id, [])
        episodes = []

        for episode_id in episode_ids[-limit:]:
            episode = self._episodes.get(episode_id)
            if episode:
                episodes.append(episode)

        return episodes

    async def search_by_tags(self, tags: list[str], limit: int = 10) -> list[Episode]:
        """Search episodes by tags."""
        if not tags:
            return []

        # Find intersection of tag sets
        matching_ids = None
        for tag in tags:
            tag_episodes = self._tag_index.get(tag, set())
            if matching_ids is None:
                matching_ids = tag_episodes.copy()
            else:
                matching_ids &= tag_episodes

        if not matching_ids:
            return []

        episodes = []
        for episode_id in list(matching_ids)[:limit]:
            episode = self._episodes.get(episode_id)
            if episode:
                episodes.append(episode)

        return episodes

    async def sample_for_replay(
        self,
        n: int = 5,
        prioritized: bool = True,
        agent_id: Optional[str] = None,
    ) -> list[Episode]:
        """
        Sample episodes for experience replay.

        Uses prioritized experience replay if enabled, weighting
        by importance and recency.
        """
        if agent_id:
            candidate_ids = self._agent_index.get(agent_id, [])
        else:
            candidate_ids = list(self._episodes.keys())

        if not candidate_ids:
            return []

        if prioritized:
            # Compute priorities
            priorities = []
            for episode_id in candidate_ids:
                episode = self._episodes.get(episode_id)
                if episode:
                    # Priority based on importance, recency, and TD error proxy
                    recency_bonus = 1.0
                    if episode.last_accessed:
                        days_ago = (datetime.now() - episode.last_accessed).days
                        recency_bonus = 1.0 / (1.0 + days_ago * 0.1)

                    priority = episode.importance * recency_bonus
                    priorities.append((priority, episode_id))

            # Sample with priority weighting
            priorities.sort(reverse=True)
            selected_ids = [p[1] for p in priorities[:n]]

        else:
            # Random sampling
            import random
            selected_ids = random.sample(candidate_ids, min(n, len(candidate_ids)))

        episodes = []
        for episode_id in selected_ids:
            episode = self._episodes.get(episode_id)
            if episode:
                episode.access_count += 1
                episode.last_accessed = datetime.now()
                episodes.append(episode)

        self._total_episodes_replayed += len(episodes)
        return episodes

    async def find_similar_outcomes(
        self,
        context: dict[str, Any],
        action: str,
        k: int = 5,
    ) -> list[tuple[Episode, float]]:
        """Find episodes with similar context and action to predict outcomes."""
        # Create query from context and action
        query_text = f"{json.dumps(context)} {action}"

        if not self._vector_store:
            return []

        results = await self._vector_store.search(query=query_text, k=k)

        episode_scores = []
        for result in results:
            episode = self._episodes.get(result.entry.id)
            if episode:
                episode_scores.append((episode, result.score))

        return episode_scores

    async def get_lessons_for_context(
        self,
        context: str,
        k: int = 10,
    ) -> list[str]:
        """Get relevant lessons learned for a given context."""
        similar_episodes = await self.search_similar(context, k=k)

        lessons = []
        for episode in similar_episodes:
            lessons.extend(episode.lessons_learned)

        # Deduplicate while preserving order
        seen = set()
        unique_lessons = []
        for lesson in lessons:
            if lesson not in seen:
                seen.add(lesson)
                unique_lessons.append(lesson)

        return unique_lessons

    async def update_importance(self, episode_id: str, importance_delta: float) -> bool:
        """Update episode importance (e.g., after successful replay)."""
        episode = self._episodes.get(episode_id)
        if not episode:
            return False

        episode.importance = max(0.0, min(1.0, episode.importance + importance_delta))
        return True

    async def _forget_least_important(self) -> None:
        """Remove least important episodes to make room."""
        if not self._episodes:
            return

        # Find episodes to forget
        episodes_by_importance = sorted(
            self._episodes.values(),
            key=lambda e: e.importance * (1.0 / (1.0 + (datetime.now() - e.start_time).days * self.decay_rate)),
        )

        to_remove = episodes_by_importance[:max(1, len(self._episodes) // 10)]

        for episode in to_remove:
            await self._remove_episode(episode.id)

        logger.info("episodes_forgotten", count=len(to_remove))

    async def _remove_episode(self, episode_id: str) -> None:
        """Remove an episode from all indexes."""
        episode = self._episodes.pop(episode_id, None)
        if not episode:
            return

        # Remove from temporal index
        self._temporal_index = [
            (t, eid) for t, eid in self._temporal_index if eid != episode_id
        ]

        # Remove from agent index
        if episode.agent_id in self._agent_index:
            self._agent_index[episode.agent_id] = [
                eid for eid in self._agent_index[episode.agent_id] if eid != episode_id
            ]

        # Remove from tag index
        for tag in episode.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(episode_id)

        # Remove from type index
        self._type_index[episode.episode_type] = [
            eid for eid in self._type_index[episode.episode_type] if eid != episode_id
        ]

        # Remove from vector store
        if self._vector_store:
            await self._vector_store.delete(episode_id)

    async def _consolidate(self) -> None:
        """Consolidate memories by extracting patterns and updating importance."""
        logger.debug("memory_consolidation_started")

        # Update importance based on patterns
        success_count = sum(1 for e in self._episodes.values() if e.success)
        failure_count = len(self._episodes) - success_count

        # Increase importance of minority class
        minority_type = "success" if success_count < failure_count else "failure"

        for episode in self._episodes.values():
            if (minority_type == "success" and episode.success) or \
               (minority_type == "failure" and not episode.success):
                episode.importance = min(1.0, episode.importance + 0.05)

        logger.debug("memory_consolidation_completed")

    async def _save_to_disk(self) -> None:
        """Save episodes to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        episodes_data = {
            episode_id: episode.to_dict()
            for episode_id, episode in self._episodes.items()
        }

        with open(self.storage_path / "episodes.json", "w") as f:
            json.dump(episodes_data, f)

        logger.info("episodic_memory_saved", path=str(self.storage_path), episodes=len(self._episodes))

    async def _load_from_disk(self) -> None:
        """Load episodes from disk."""
        if not self.storage_path:
            return

        episodes_path = self.storage_path / "episodes.json"

        if episodes_path.exists():
            with open(episodes_path) as f:
                episodes_data = json.load(f)

            for episode_id, data in episodes_data.items():
                episode = Episode.from_dict(data)
                self._episodes[episode_id] = episode

                # Rebuild indexes
                self._temporal_index.append((episode.start_time, episode.id))

                if episode.agent_id:
                    if episode.agent_id not in self._agent_index:
                        self._agent_index[episode.agent_id] = []
                    self._agent_index[episode.agent_id].append(episode.id)

                for tag in episode.tags:
                    if tag not in self._tag_index:
                        self._tag_index[tag] = set()
                    self._tag_index[tag].add(episode.id)

                self._type_index[episode.episode_type].append(episode.id)

            self._temporal_index.sort(key=lambda x: x[0])

        logger.info("episodic_memory_loaded", path=str(self.storage_path), episodes=len(self._episodes))

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        type_counts = {t.value: len(ids) for t, ids in self._type_index.items()}

        return {
            "total_episodes": len(self._episodes),
            "max_episodes": self.max_episodes,
            "total_stored": self._total_episodes_stored,
            "total_replayed": self._total_episodes_replayed,
            "by_type": type_counts,
            "agents_tracked": len(self._agent_index),
            "unique_tags": len(self._tag_index),
            "avg_importance": sum(e.importance for e in self._episodes.values()) / max(1, len(self._episodes)),
        }
