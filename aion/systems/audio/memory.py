"""
AION Audio Memory

Audio memory system for storing and retrieving audio experiences:
- Audio similarity search using FAISS/embeddings
- Speaker-based retrieval
- Transcript search
- Conversation tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioScene,
    AudioMemoryEntry,
    AudioSearchResult,
    Speaker,
    Transcript,
    VoiceProfile,
    TimeRange,
)

logger = structlog.get_logger(__name__)


@dataclass
class ConversationEntry:
    """An entry in conversation memory."""
    id: str
    transcript: Transcript
    speakers: list[Speaker]
    topic: Optional[str]
    context: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    importance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "transcript": self.transcript.to_dict(),
            "speakers": [s.to_dict() for s in self.speakers],
            "topic": self.topic,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "importance": self.importance,
        }


class AudioMemory:
    """
    AION Audio Memory System

    Stores audio experiences and enables retrieval by:
    - Audio embedding similarity (CLAP)
    - Speaker matching
    - Transcript text search
    - Temporal queries
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        max_entries: int = 10000,
        persistence_path: Optional[Path] = None,
        use_faiss: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.persistence_path = persistence_path
        self.use_faiss = use_faiss

        # Storage
        self._entries: dict[str, AudioMemoryEntry] = {}
        self._embeddings: list[np.ndarray] = []
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}

        # FAISS index (optional)
        self._faiss_index = None
        self._faiss_available = False

        # Speaker index
        self._speaker_index: dict[str, set[str]] = {}  # speaker_id -> memory IDs

        # Transcript text index (simple inverted index)
        self._text_index: dict[str, set[str]] = {}  # word -> memory IDs

        # Voice profiles for speaker identification
        self._voice_profiles: dict[str, VoiceProfile] = {}

        # Conversation memory
        self._conversations: dict[str, ConversationEntry] = {}

        # Statistics
        self._stats = {
            "entries_stored": 0,
            "searches_performed": 0,
            "speaker_searches": 0,
            "text_searches": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the audio memory system."""
        if self._initialized:
            return

        logger.info("Initializing Audio Memory System")

        # Try to initialize FAISS
        if self.use_faiss:
            await self._initialize_faiss()

        # Load persisted data if available
        if self.persistence_path and self.persistence_path.exists():
            await self._load_from_disk()

        self._initialized = True
        logger.info(
            "Audio Memory System initialized",
            faiss_enabled=self._faiss_available,
            entries=len(self._entries),
        )

    async def _initialize_faiss(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss

            # Use IndexFlatIP for cosine similarity (normalized vectors)
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self._faiss_available = True
            logger.info("FAISS index initialized")

        except ImportError:
            logger.warning("FAISS not available, using numpy-based similarity search")
            self._faiss_available = False

    async def shutdown(self) -> None:
        """Shutdown the memory system."""
        if self.persistence_path:
            await self._save_to_disk()

        self._initialized = False
        logger.info("Audio Memory System shutdown")

    async def _save_to_disk(self) -> None:
        """Save memory to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)

            # Save entries (without waveforms)
            entries_data = {}
            for entry_id, entry in self._entries.items():
                entries_data[entry_id] = {
                    "id": entry.id,
                    "audio_hash": entry.audio_hash,
                    "embedding": entry.embedding.tolist(),
                    "scene": entry.scene.to_dict(),
                    "transcript_text": entry.transcript_text,
                    "duration": entry.duration,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at.isoformat(),
                    "access_count": entry.access_count,
                    "importance": entry.importance,
                }

            with open(self.persistence_path / "entries.json", "w") as f:
                json.dump(entries_data, f)

            # Save voice profiles
            profiles_data = {}
            for profile_id, profile in self._voice_profiles.items():
                profiles_data[profile_id] = {
                    "id": profile.id,
                    "name": profile.name,
                    "embedding": profile.embedding.tolist(),
                    "sample_count": profile.sample_count,
                    "total_duration": profile.total_duration,
                    "metadata": profile.metadata,
                }

            with open(self.persistence_path / "voice_profiles.json", "w") as f:
                json.dump(profiles_data, f)

            # Save FAISS index
            if self._faiss_available and self._faiss_index.ntotal > 0:
                import faiss
                faiss.write_index(
                    self._faiss_index,
                    str(self.persistence_path / "audio_index.faiss")
                )

            logger.info("Audio memory saved to disk", entries=len(self._entries))

        except Exception as e:
            logger.error(f"Failed to save audio memory: {e}")

    async def _load_from_disk(self) -> None:
        """Load memory from disk."""
        if not self.persistence_path:
            return

        try:
            # Load entries
            entries_path = self.persistence_path / "entries.json"
            if entries_path.exists():
                with open(entries_path) as f:
                    entries_data = json.load(f)

                for entry_data in entries_data.values():
                    # Reconstruct minimal AudioScene
                    scene = self._reconstruct_scene(entry_data["scene"])

                    entry = AudioMemoryEntry(
                        id=entry_data["id"],
                        audio_hash=entry_data["audio_hash"],
                        embedding=np.array(entry_data["embedding"], dtype=np.float32),
                        scene=scene,
                        transcript_text=entry_data.get("transcript_text"),
                        duration=entry_data["duration"],
                        metadata=entry_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(entry_data["created_at"]),
                        access_count=entry_data.get("access_count", 0),
                        importance=entry_data.get("importance", 0.5),
                    )

                    self._entries[entry.id] = entry
                    self._add_to_indexes(entry)

            # Load voice profiles
            profiles_path = self.persistence_path / "voice_profiles.json"
            if profiles_path.exists():
                with open(profiles_path) as f:
                    profiles_data = json.load(f)

                for profile_data in profiles_data.values():
                    profile = VoiceProfile(
                        id=profile_data["id"],
                        name=profile_data["name"],
                        embedding=np.array(profile_data["embedding"], dtype=np.float32),
                        sample_count=profile_data.get("sample_count", 1),
                        total_duration=profile_data.get("total_duration", 0),
                        metadata=profile_data.get("metadata", {}),
                    )
                    self._voice_profiles[profile.id] = profile

            # Load FAISS index
            if self._faiss_available:
                index_path = self.persistence_path / "audio_index.faiss"
                if index_path.exists():
                    import faiss
                    self._faiss_index = faiss.read_index(str(index_path))

            logger.info("Audio memory loaded from disk", entries=len(self._entries))

        except Exception as e:
            logger.error(f"Failed to load audio memory: {e}")

    def _reconstruct_scene(self, scene_data: dict) -> AudioScene:
        """Reconstruct AudioScene from dict (minimal)."""
        from aion.systems.audio.models import AudioEvent, AudioRelation

        events = []
        for e in scene_data.get("events", []):
            events.append(AudioEvent(
                id=e["id"],
                label=e["label"],
                category=e.get("category", "other"),
                start_time=e["start_time"],
                end_time=e["end_time"],
                confidence=e["confidence"],
            ))

        speakers = []
        for s in scene_data.get("speakers", []):
            speakers.append(Speaker(
                id=s["id"],
                name=s.get("name"),
                segments=[TimeRange(seg["start"], seg["end"]) for seg in s.get("segments", [])],
                confidence=s.get("confidence", 0),
            ))

        return AudioScene(
            id=scene_data["id"],
            audio_id=scene_data["audio_id"],
            duration=scene_data["duration"],
            events=events,
            speakers=speakers,
            ambient_description=scene_data.get("ambient_description", ""),
            emotional_tone=scene_data.get("emotional_tone"),
            noise_level_db=scene_data.get("noise_level_db", 0),
        )

    def _add_to_indexes(self, entry: AudioMemoryEntry) -> None:
        """Add entry to all indexes."""
        # Add to embedding index
        idx = len(self._embeddings)
        self._embeddings.append(entry.embedding)
        self._id_to_idx[entry.id] = idx
        self._idx_to_id[idx] = entry.id

        # Add to FAISS if available
        if self._faiss_available:
            # Normalize embedding for cosine similarity
            normalized = entry.embedding / (np.linalg.norm(entry.embedding) + 1e-8)
            self._faiss_index.add(normalized.reshape(1, -1))

        # Add to speaker index
        for speaker in entry.scene.speakers:
            if speaker.id not in self._speaker_index:
                self._speaker_index[speaker.id] = set()
            self._speaker_index[speaker.id].add(entry.id)

        # Add to text index
        if entry.transcript_text:
            words = set(entry.transcript_text.lower().split())
            for word in words:
                if len(word) > 2:  # Skip very short words
                    if word not in self._text_index:
                        self._text_index[word] = set()
                    self._text_index[word].add(entry.id)

    async def store(
        self,
        scene: AudioScene,
        embedding: np.ndarray,
        transcript_text: Optional[str] = None,
        audio_hash: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> AudioMemoryEntry:
        """
        Store an audio memory.

        Args:
            scene: Audio scene analysis
            embedding: Audio embedding vector
            transcript_text: Optional transcript text
            audio_hash: Optional hash for deduplication
            metadata: Additional metadata
            importance: Importance score (0-1)

        Returns:
            Created AudioMemoryEntry
        """
        if not self._initialized:
            await self.initialize()

        # Check for duplicate
        if audio_hash:
            for existing in self._entries.values():
                if existing.audio_hash == audio_hash:
                    logger.debug("Duplicate audio detected", hash=audio_hash)
                    existing.access_count += 1
                    existing.last_accessed = datetime.now()
                    return existing

        # Create entry
        entry = AudioMemoryEntry(
            id=str(uuid.uuid4()),
            audio_hash=audio_hash or "",
            embedding=embedding.astype(np.float32),
            scene=scene,
            transcript_text=transcript_text,
            duration=scene.duration,
            metadata=metadata or {},
            importance=importance,
        )

        # Store entry
        self._entries[entry.id] = entry
        self._add_to_indexes(entry)

        self._stats["entries_stored"] += 1

        # Enforce max entries
        if len(self._entries) > self.max_entries:
            await self._evict_oldest()

        logger.debug("Audio memory stored", entry_id=entry.id, duration=scene.duration)
        return entry

    async def _evict_oldest(self) -> None:
        """Evict oldest/least important entries."""
        # Sort by importance, access count, and recency
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.access_count, e.created_at),
        )

        # Remove oldest 10%
        to_remove = sorted_entries[:max(1, len(sorted_entries) // 10)]

        for entry in to_remove:
            await self._remove_entry(entry.id)

    async def _remove_entry(self, entry_id: str) -> None:
        """Remove an entry from all indexes."""
        if entry_id not in self._entries:
            return

        entry = self._entries[entry_id]

        # Remove from speaker index
        for speaker in entry.scene.speakers:
            if speaker.id in self._speaker_index:
                self._speaker_index[speaker.id].discard(entry_id)

        # Remove from text index
        if entry.transcript_text:
            words = set(entry.transcript_text.lower().split())
            for word in words:
                if word in self._text_index:
                    self._text_index[word].discard(entry_id)

        # Note: We don't remove from embedding list for efficiency
        # FAISS index would need rebuilding periodically

        del self._entries[entry_id]

    async def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[AudioSearchResult]:
        """
        Search for similar audio by embedding.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of AudioSearchResult sorted by similarity
        """
        if not self._entries:
            return []

        self._stats["searches_performed"] += 1

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        if self._faiss_available and self._faiss_index.ntotal > 0:
            # Use FAISS for fast search
            similarities, indices = self._faiss_index.search(
                query_norm.reshape(1, -1).astype(np.float32),
                min(limit * 2, self._faiss_index.ntotal),  # Get more than needed for filtering
            )

            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < 0 or idx >= len(self._idx_to_id):
                    continue

                entry_id = self._idx_to_id.get(idx)
                if entry_id and entry_id in self._entries:
                    if sim >= min_similarity:
                        results.append(AudioSearchResult(
                            entry=self._entries[entry_id],
                            similarity=float(sim),
                            match_type="embedding",
                        ))

            # Update access tracking
            for result in results[:limit]:
                result.entry.access_count += 1
                result.entry.last_accessed = datetime.now()

            return results[:limit]

        else:
            # Numpy-based search
            results = []
            for entry_id, entry in self._entries.items():
                entry_norm = entry.embedding / (np.linalg.norm(entry.embedding) + 1e-8)
                similarity = float(np.dot(query_norm, entry_norm))

                if similarity >= min_similarity:
                    results.append(AudioSearchResult(
                        entry=entry,
                        similarity=similarity,
                        match_type="embedding",
                    ))

            results.sort(key=lambda r: r.similarity, reverse=True)

            # Update access tracking
            for result in results[:limit]:
                result.entry.access_count += 1
                result.entry.last_accessed = datetime.now()

            return results[:limit]

    async def search_by_text(
        self,
        query: str,
        limit: int = 10,
    ) -> list[AudioSearchResult]:
        """
        Search for audio by transcript text.

        Args:
            query: Text query
            limit: Maximum results

        Returns:
            List of AudioSearchResult
        """
        self._stats["text_searches"] += 1

        query_words = set(query.lower().split())

        # Find matching entries
        entry_scores: dict[str, float] = {}

        for word in query_words:
            if word in self._text_index:
                for entry_id in self._text_index[word]:
                    if entry_id in self._entries:
                        entry_scores[entry_id] = entry_scores.get(entry_id, 0) + 1

        # Score by word overlap
        results = []
        for entry_id, score in entry_scores.items():
            similarity = score / len(query_words)
            results.append(AudioSearchResult(
                entry=self._entries[entry_id],
                similarity=similarity,
                match_type="transcript",
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)

        return results[:limit]

    async def search_by_speaker(
        self,
        speaker: Union[Speaker, VoiceProfile, str],
        limit: int = 10,
    ) -> list[AudioSearchResult]:
        """
        Search for audio containing a specific speaker.

        Args:
            speaker: Speaker to search for (object, profile, or ID)
            limit: Maximum results

        Returns:
            List of AudioSearchResult
        """
        self._stats["speaker_searches"] += 1

        # Get speaker ID
        if isinstance(speaker, str):
            speaker_id = speaker
        elif isinstance(speaker, VoiceProfile):
            # Search by voice profile embedding
            return await self._search_by_voice_embedding(speaker.embedding, limit)
        else:
            speaker_id = speaker.id

        # Find entries with this speaker
        matching_ids = self._speaker_index.get(speaker_id, set())

        results = []
        for entry_id in matching_ids:
            if entry_id in self._entries:
                entry = self._entries[entry_id]
                # Calculate speaker's proportion of audio
                speaker_time = sum(
                    s.total_speaking_time for s in entry.scene.speakers
                    if s.id == speaker_id
                )
                similarity = speaker_time / entry.duration if entry.duration > 0 else 0

                results.append(AudioSearchResult(
                    entry=entry,
                    similarity=similarity,
                    match_type="speaker",
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    async def _search_by_voice_embedding(
        self,
        voice_embedding: np.ndarray,
        limit: int,
    ) -> list[AudioSearchResult]:
        """Search by matching voice embedding against stored speakers."""
        results = []

        voice_norm = voice_embedding / (np.linalg.norm(voice_embedding) + 1e-8)

        for entry in self._entries.values():
            best_match = 0.0
            for speaker in entry.scene.speakers:
                if speaker.embedding is not None:
                    spk_norm = speaker.embedding / (np.linalg.norm(speaker.embedding) + 1e-8)
                    similarity = float(np.dot(voice_norm, spk_norm))
                    best_match = max(best_match, similarity)

            if best_match > 0.5:  # Threshold for voice match
                results.append(AudioSearchResult(
                    entry=entry,
                    similarity=best_match,
                    match_type="speaker",
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    async def search_by_time_range(
        self,
        start: datetime,
        end: datetime,
        limit: int = 10,
    ) -> list[AudioSearchResult]:
        """
        Search for audio within a time range.

        Args:
            start: Start datetime
            end: End datetime
            limit: Maximum results

        Returns:
            List of AudioSearchResult
        """
        results = []

        for entry in self._entries.values():
            if start <= entry.created_at <= end:
                # Score by recency within range
                time_diff = (end - entry.created_at).total_seconds()
                total_range = (end - start).total_seconds()
                similarity = 1 - (time_diff / total_range) if total_range > 0 else 1

                results.append(AudioSearchResult(
                    entry=entry,
                    similarity=similarity,
                    match_type="temporal",
                ))

        results.sort(key=lambda r: r.entry.created_at, reverse=True)
        return results[:limit]

    # ==================== Voice Profile Management ====================

    async def register_voice_profile(
        self,
        name: str,
        embedding: np.ndarray,
        duration: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> VoiceProfile:
        """
        Register a new voice profile.

        Args:
            name: Speaker name
            embedding: Voice embedding
            duration: Duration of audio used
            metadata: Additional metadata

        Returns:
            Created VoiceProfile
        """
        profile = VoiceProfile(
            id=str(uuid.uuid4()),
            name=name,
            embedding=embedding.astype(np.float32),
            total_duration=duration,
            metadata=metadata or {},
        )

        self._voice_profiles[profile.id] = profile
        logger.info("Voice profile registered", name=name, profile_id=profile.id)

        return profile

    async def update_voice_profile(
        self,
        profile_id: str,
        embedding: np.ndarray,
        duration: float,
    ) -> Optional[VoiceProfile]:
        """
        Update voice profile with new sample.

        Args:
            profile_id: Profile ID
            embedding: New voice embedding
            duration: Duration of new sample

        Returns:
            Updated profile or None
        """
        if profile_id not in self._voice_profiles:
            return None

        profile = self._voice_profiles[profile_id]
        profile.update_embedding(embedding, duration)

        logger.debug("Voice profile updated", profile_id=profile_id)
        return profile

    async def identify_voice(
        self,
        embedding: np.ndarray,
        threshold: float = 0.7,
    ) -> tuple[Optional[VoiceProfile], float]:
        """
        Identify voice against registered profiles.

        Args:
            embedding: Voice embedding to match
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (matched_profile, similarity) or (None, 0)
        """
        if not self._voice_profiles:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for profile in self._voice_profiles.values():
            similarity = profile.similarity(embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = profile

        if best_similarity >= threshold:
            return best_match, best_similarity

        return None, best_similarity

    def get_voice_profiles(self) -> list[VoiceProfile]:
        """Get all registered voice profiles."""
        return list(self._voice_profiles.values())

    def get_voice_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get a specific voice profile."""
        return self._voice_profiles.get(profile_id)

    # ==================== Conversation Memory ====================

    async def store_conversation(
        self,
        transcript: Transcript,
        topic: Optional[str] = None,
        context: Optional[dict] = None,
        importance: float = 0.5,
    ) -> ConversationEntry:
        """
        Store a conversation in memory.

        Args:
            transcript: Conversation transcript
            topic: Conversation topic
            context: Additional context
            importance: Importance score

        Returns:
            Created ConversationEntry
        """
        entry = ConversationEntry(
            id=str(uuid.uuid4()),
            transcript=transcript,
            speakers=transcript.speakers,
            topic=topic,
            context=context or {},
            importance=importance,
        )

        self._conversations[entry.id] = entry
        logger.debug("Conversation stored", entry_id=entry.id, speakers=len(transcript.speakers))

        return entry

    async def search_conversations(
        self,
        query: Optional[str] = None,
        speaker: Optional[Union[Speaker, str]] = None,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> list[ConversationEntry]:
        """
        Search conversation memory.

        Args:
            query: Text query
            speaker: Speaker to filter by
            topic: Topic to filter by
            limit: Maximum results

        Returns:
            List of matching conversations
        """
        results = []

        speaker_id = speaker.id if isinstance(speaker, Speaker) else speaker

        for conv in self._conversations.values():
            score = 0.0

            # Match by query text
            if query:
                query_words = set(query.lower().split())
                text_words = set(conv.transcript.text.lower().split())
                overlap = len(query_words & text_words)
                if overlap > 0:
                    score += overlap / len(query_words)

            # Match by speaker
            if speaker_id:
                if any(s.id == speaker_id for s in conv.speakers):
                    score += 0.5

            # Match by topic
            if topic and conv.topic:
                if topic.lower() in conv.topic.lower():
                    score += 0.5

            if score > 0:
                results.append((conv, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]

    # ==================== Utilities ====================

    async def get(self, entry_id: str) -> Optional[AudioMemoryEntry]:
        """Get a specific memory entry."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        return entry

    def count(self) -> int:
        """Get number of stored memories."""
        return len(self._entries)

    def count_voice_profiles(self) -> int:
        """Get number of registered voice profiles."""
        return len(self._voice_profiles)

    def count_conversations(self) -> int:
        """Get number of stored conversations."""
        return len(self._conversations)

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            **self._stats,
            "total_entries": len(self._entries),
            "voice_profiles": len(self._voice_profiles),
            "conversations": len(self._conversations),
            "unique_speakers": len(self._speaker_index),
            "indexed_words": len(self._text_index),
            "faiss_enabled": self._faiss_available,
        }
