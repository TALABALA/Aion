"""
AION Audio Memory

Audio memory system for storing and retrieving audio experiences:
- FAISS-powered similarity search using CLAP embeddings
- Speaker-based retrieval
- Transcript/content-based search
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
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioMemoryEntry,
    AudioScene,
    Speaker,
    Transcript,
)

logger = structlog.get_logger(__name__)


@dataclass
class AudioSearchResult:
    """Result of an audio memory search."""
    entry: AudioMemoryEntry
    similarity: float
    match_type: str = "embedding"  # "embedding", "speaker", "transcript", "tag"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "similarity": self.similarity,
            "match_type": self.match_type,
        }


@dataclass
class ConversationSegment:
    """A segment of a tracked conversation."""
    memory_id: str
    speaker_id: Optional[str]
    text: str
    start_time: float
    end_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Conversation:
    """A tracked conversation across multiple audio entries."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    segments: list[ConversationSegment] = field(default_factory=list)
    speakers: dict[str, str] = field(default_factory=dict)  # speaker_id -> name
    topics: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def full_transcript(self) -> str:
        return " ".join(seg.text for seg in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "segment_count": len(self.segments),
            "speakers": self.speakers,
            "topics": self.topics,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class AudioMemory:
    """
    AION Audio Memory System

    Stores audio experiences and enables retrieval by:
    - Acoustic similarity (CLAP embeddings via FAISS)
    - Speaker identity
    - Transcript content
    - Tags and metadata
    - Conversation tracking
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        max_entries: int = 50000,
        index_path: Optional[Union[str, Path]] = None,
        use_gpu: bool = False,
    ):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.index_path = Path(index_path) if index_path else None
        self.use_gpu = use_gpu

        # Storage
        self._entries: dict[str, AudioMemoryEntry] = {}
        self._faiss_index = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._next_idx: int = 0

        # Indexes for different search types
        self._speaker_index: dict[str, set[str]] = {}  # speaker_embedding_hash -> memory IDs
        self._tag_index: dict[str, set[str]] = {}  # tag -> memory IDs
        self._transcript_index: dict[str, set[str]] = {}  # word -> memory IDs

        # Conversation tracking
        self._conversations: dict[str, Conversation] = {}
        self._active_conversation: Optional[str] = None

        # Registered speakers
        self._registered_speakers: dict[str, Speaker] = {}

        # Statistics
        self._stats = {
            "entries_stored": 0,
            "entries_retrieved": 0,
            "searches_performed": 0,
            "conversations_tracked": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the audio memory system."""
        if self._initialized:
            return

        logger.info("Initializing Audio Memory System")

        # Initialize FAISS index
        self._init_faiss_index()

        # Load from disk if path exists
        if self.index_path and self.index_path.exists():
            await self._load_from_disk()

        self._initialized = True
        logger.info("Audio Memory System initialized", entries=len(self._entries))

    def _init_faiss_index(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss

            # Use IVF index for large-scale search
            if self.max_entries > 10000:
                # IVF with flat quantizer
                quantizer = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine sim
                nlist = min(100, max(10, self.max_entries // 100))
                self._faiss_index = faiss.IndexIVFFlat(
                    quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                # Simple flat index for smaller collections
                self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self._faiss_index = faiss.index_cpu_to_gpu(res, 0, self._faiss_index)
                    logger.info("Using GPU-accelerated FAISS")
                except Exception as e:
                    logger.warning(f"GPU FAISS not available: {e}")

            logger.info("FAISS index initialized", dim=self.embedding_dim)

        except ImportError:
            logger.warning("FAISS not available, using fallback search")
            self._faiss_index = None

    async def shutdown(self) -> None:
        """Shutdown and persist memory."""
        if self.index_path:
            await self._save_to_disk()

        self._initialized = False
        logger.info("Audio Memory System shutdown")

    async def _save_to_disk(self) -> None:
        """Save memory to disk."""
        if not self.index_path:
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save entries
        entries_path = self.index_path.with_suffix(".entries.pkl")
        with open(entries_path, "wb") as f:
            # Can't pickle numpy arrays directly in entries, so convert
            serializable_entries = {}
            for entry_id, entry in self._entries.items():
                entry_dict = {
                    "id": entry.id,
                    "audio_hash": entry.audio_hash,
                    "embedding": entry.embedding.tolist() if entry.embedding is not None else None,
                    "audio_scene": entry.audio_scene.to_dict() if entry.audio_scene else None,
                    "transcript": entry.transcript.to_dict() if entry.transcript else None,
                    "context": entry.context,
                    "source_path": entry.source_path,
                    "duration": entry.duration,
                    "importance": entry.importance,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                    "access_count": entry.access_count,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                }
                serializable_entries[entry_id] = entry_dict
            pickle.dump(serializable_entries, f)

        # Save FAISS index
        if self._faiss_index is not None:
            import faiss
            faiss_path = self.index_path.with_suffix(".faiss")
            faiss.write_index(faiss.index_gpu_to_cpu(self._faiss_index) if self.use_gpu else self._faiss_index, str(faiss_path))

        # Save indexes
        indexes_path = self.index_path.with_suffix(".indexes.json")
        with open(indexes_path, "w") as f:
            json.dump({
                "id_to_idx": self._id_to_idx,
                "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
                "speaker_index": {k: list(v) for k, v in self._speaker_index.items()},
                "tag_index": {k: list(v) for k, v in self._tag_index.items()},
                "next_idx": self._next_idx,
            }, f)

        # Save registered speakers
        speakers_path = self.index_path.with_suffix(".speakers.pkl")
        with open(speakers_path, "wb") as f:
            pickle.dump(self._registered_speakers, f)

        logger.info("Audio memory saved to disk", path=str(self.index_path))

    async def _load_from_disk(self) -> None:
        """Load memory from disk."""
        try:
            # Load entries
            entries_path = self.index_path.with_suffix(".entries.pkl")
            if entries_path.exists():
                with open(entries_path, "rb") as f:
                    serializable_entries = pickle.load(f)

                for entry_id, entry_dict in serializable_entries.items():
                    self._entries[entry_id] = AudioMemoryEntry(
                        id=entry_dict["id"],
                        audio_hash=entry_dict["audio_hash"],
                        embedding=np.array(entry_dict["embedding"]) if entry_dict["embedding"] else None,
                        context=entry_dict["context"],
                        source_path=entry_dict["source_path"],
                        duration=entry_dict["duration"],
                        importance=entry_dict["importance"],
                        created_at=datetime.fromisoformat(entry_dict["created_at"]),
                        last_accessed=datetime.fromisoformat(entry_dict["last_accessed"]) if entry_dict["last_accessed"] else None,
                        access_count=entry_dict["access_count"],
                        tags=entry_dict["tags"],
                        metadata=entry_dict["metadata"],
                    )

            # Load FAISS index
            faiss_path = self.index_path.with_suffix(".faiss")
            if faiss_path.exists() and self._faiss_index is not None:
                import faiss
                self._faiss_index = faiss.read_index(str(faiss_path))
                if self.use_gpu:
                    try:
                        res = faiss.StandardGpuResources()
                        self._faiss_index = faiss.index_cpu_to_gpu(res, 0, self._faiss_index)
                    except Exception:
                        pass

            # Load indexes
            indexes_path = self.index_path.with_suffix(".indexes.json")
            if indexes_path.exists():
                with open(indexes_path, "r") as f:
                    indexes = json.load(f)
                    self._id_to_idx = indexes["id_to_idx"]
                    self._idx_to_id = {int(k): v for k, v in indexes["idx_to_id"].items()}
                    self._speaker_index = {k: set(v) for k, v in indexes["speaker_index"].items()}
                    self._tag_index = {k: set(v) for k, v in indexes["tag_index"].items()}
                    self._next_idx = indexes["next_idx"]

            # Load registered speakers
            speakers_path = self.index_path.with_suffix(".speakers.pkl")
            if speakers_path.exists():
                with open(speakers_path, "rb") as f:
                    self._registered_speakers = pickle.load(f)

            logger.info("Audio memory loaded from disk", entries=len(self._entries))

        except Exception as e:
            logger.error(f"Failed to load audio memory: {e}")

    def compute_audio_hash(self, waveform: np.ndarray) -> str:
        """Compute hash for audio deduplication."""
        return hashlib.sha256(waveform.tobytes()).hexdigest()[:16]

    async def store(
        self,
        embedding: np.ndarray,
        audio_scene: Optional[AudioScene] = None,
        transcript: Optional[Transcript] = None,
        context: str = "",
        source_path: Optional[str] = None,
        duration: float = 0.0,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        audio_hash: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AudioMemoryEntry:
        """
        Store an audio memory.

        Args:
            embedding: Audio embedding (CLAP or similar)
            audio_scene: Full scene analysis
            transcript: Speech transcription
            context: User-provided context
            source_path: Original file path
            duration: Audio duration in seconds
            importance: Importance score (0-1)
            tags: Organizational tags
            audio_hash: Hash for deduplication
            metadata: Additional metadata

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

        # Normalize embedding
        embedding = self._normalize_embedding(embedding)

        # Create entry
        entry = AudioMemoryEntry(
            audio_hash=audio_hash or "",
            embedding=embedding,
            audio_scene=audio_scene,
            transcript=transcript,
            context=context,
            source_path=source_path,
            duration=duration,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Store entry
        self._entries[entry.id] = entry

        # Add to FAISS index
        if self._faiss_index is not None:
            self._add_to_faiss(entry.id, embedding)
        else:
            # Fallback: store mapping for brute-force search
            self._id_to_idx[entry.id] = self._next_idx
            self._idx_to_id[self._next_idx] = entry.id
            self._next_idx += 1

        # Update auxiliary indexes
        self._update_indexes(entry)

        self._stats["entries_stored"] += 1

        # Enforce max entries
        if len(self._entries) > self.max_entries:
            await self._evict_entries()

        logger.debug("Audio memory stored", entry_id=entry.id)
        return entry

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        embedding = embedding.astype(np.float32)

        # Ensure correct dimension
        if embedding.shape[0] != self.embedding_dim:
            if embedding.shape[0] < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - embedding.shape[0]))
            else:
                embedding = embedding[:self.embedding_dim]

        # L2 normalize for cosine similarity via inner product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _add_to_faiss(self, entry_id: str, embedding: np.ndarray) -> None:
        """Add embedding to FAISS index."""
        import faiss

        # Train index if needed (for IVF)
        if hasattr(self._faiss_index, "is_trained") and not self._faiss_index.is_trained:
            # Need to train with some vectors first
            if len(self._entries) >= 100:
                all_embeddings = np.stack([
                    e.embedding for e in self._entries.values()
                    if e.embedding is not None
                ]).astype(np.float32)
                self._faiss_index.train(all_embeddings)

        # Add to index
        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        self._faiss_index.add(embedding_2d)

        # Update mapping
        idx = self._next_idx
        self._id_to_idx[entry_id] = idx
        self._idx_to_id[idx] = entry_id
        self._next_idx += 1

    def _update_indexes(self, entry: AudioMemoryEntry) -> None:
        """Update auxiliary indexes for an entry."""
        # Tag index
        for tag in entry.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(entry.id)

        # Transcript index (simple word-based)
        if entry.transcript:
            words = entry.transcript.text.lower().split()
            for word in set(words):
                if len(word) > 2:  # Skip very short words
                    if word not in self._transcript_index:
                        self._transcript_index[word] = set()
                    self._transcript_index[word].add(entry.id)

        # Speaker index (based on speaker embeddings)
        if entry.audio_scene and entry.audio_scene.speakers:
            for speaker in entry.audio_scene.speakers:
                if speaker.embedding is not None:
                    speaker_hash = hashlib.sha256(speaker.embedding.tobytes()).hexdigest()[:16]
                    if speaker_hash not in self._speaker_index:
                        self._speaker_index[speaker_hash] = set()
                    self._speaker_index[speaker_hash].add(entry.id)

    async def _evict_entries(self) -> None:
        """Evict least important/accessed entries."""
        # Sort by importance and recency
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.access_count, e.created_at),
        )

        # Remove oldest 10%
        to_remove = sorted_entries[:len(sorted_entries) // 10]

        for entry in to_remove:
            await self.delete(entry.id)

        logger.info("Evicted entries", count=len(to_remove))

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id not in self._entries:
            return False

        entry = self._entries[entry_id]

        # Remove from indexes
        for tag in entry.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(entry_id)

        if entry.transcript:
            words = entry.transcript.text.lower().split()
            for word in set(words):
                if word in self._transcript_index:
                    self._transcript_index[word].discard(entry_id)

        # Note: FAISS doesn't support deletion, so we mark as deleted
        # In production, would periodically rebuild index
        del self._entries[entry_id]

        return True

    async def search_by_similarity(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[AudioSearchResult]:
        """
        Search for similar audio memories using embeddings.

        Args:
            query_embedding: Query embedding (CLAP or similar)
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of AudioSearchResult sorted by similarity
        """
        if not self._entries:
            return []

        query_embedding = self._normalize_embedding(query_embedding)
        self._stats["searches_performed"] += 1

        if self._faiss_index is not None:
            return self._search_faiss(query_embedding, limit, min_similarity)
        else:
            return self._search_brute_force(query_embedding, limit, min_similarity)

    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        limit: int,
        min_similarity: float,
    ) -> list[AudioSearchResult]:
        """Search using FAISS index."""
        query_2d = query_embedding.reshape(1, -1).astype(np.float32)

        # Handle untrained IVF index
        if hasattr(self._faiss_index, "is_trained") and not self._faiss_index.is_trained:
            return self._search_brute_force(query_embedding, limit, min_similarity)

        # Search
        similarities, indices = self._faiss_index.search(query_2d, min(limit * 2, len(self._entries)))

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue

            entry_id = self._idx_to_id.get(int(idx))
            if entry_id and entry_id in self._entries:
                if sim >= min_similarity:
                    results.append(AudioSearchResult(
                        entry=self._entries[entry_id],
                        similarity=float(sim),
                        match_type="embedding",
                    ))
                    self._entries[entry_id].access_count += 1
                    self._entries[entry_id].last_accessed = datetime.now()

        return results[:limit]

    def _search_brute_force(
        self,
        query_embedding: np.ndarray,
        limit: int,
        min_similarity: float,
    ) -> list[AudioSearchResult]:
        """Brute-force search when FAISS is not available."""
        results = []

        for entry in self._entries.values():
            if entry.embedding is None:
                continue

            similarity = float(np.dot(query_embedding, entry.embedding))

            if similarity >= min_similarity:
                results.append(AudioSearchResult(
                    entry=entry,
                    similarity=similarity,
                    match_type="embedding",
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)

        for result in results[:limit]:
            result.entry.access_count += 1
            result.entry.last_accessed = datetime.now()

        return results[:limit]

    async def search_by_text(
        self,
        text_embedding: np.ndarray,
        limit: int = 10,
    ) -> list[AudioSearchResult]:
        """
        Search for audio using text query embedding (CLAP text encoder).

        Args:
            text_embedding: Text embedding from CLAP
            limit: Maximum results

        Returns:
            List of AudioSearchResult
        """
        return await self.search_by_similarity(text_embedding, limit)

    async def search_by_transcript(
        self,
        query: str,
        limit: int = 10,
    ) -> list[AudioSearchResult]:
        """
        Search for audio by transcript content.

        Args:
            query: Text query
            limit: Maximum results

        Returns:
            List of AudioSearchResult
        """
        query_words = set(query.lower().split())
        matching_ids: dict[str, int] = {}  # id -> word match count

        for word in query_words:
            if len(word) > 2 and word in self._transcript_index:
                for entry_id in self._transcript_index[word]:
                    matching_ids[entry_id] = matching_ids.get(entry_id, 0) + 1

        results = []
        for entry_id, match_count in matching_ids.items():
            if entry_id in self._entries:
                entry = self._entries[entry_id]
                # Score by word match ratio
                if entry.transcript:
                    total_words = len(entry.transcript.text.split())
                    similarity = match_count / max(len(query_words), 1)
                    results.append(AudioSearchResult(
                        entry=entry,
                        similarity=similarity,
                        match_type="transcript",
                    ))

        results.sort(key=lambda r: r.similarity, reverse=True)

        self._stats["searches_performed"] += 1

        return results[:limit]

    async def search_by_speaker(
        self,
        speaker_embedding: np.ndarray,
        limit: int = 10,
        similarity_threshold: float = 0.75,
    ) -> list[AudioSearchResult]:
        """
        Search for audio by speaker.

        Args:
            speaker_embedding: Speaker embedding
            limit: Maximum results
            similarity_threshold: Minimum speaker similarity

        Returns:
            List of AudioSearchResult
        """
        results = []

        for entry in self._entries.values():
            if entry.audio_scene and entry.audio_scene.speakers:
                for speaker in entry.audio_scene.speakers:
                    if speaker.embedding is not None:
                        # Cosine similarity
                        sim = float(np.dot(speaker_embedding, speaker.embedding) /
                                   (np.linalg.norm(speaker_embedding) * np.linalg.norm(speaker.embedding) + 1e-10))
                        if sim >= similarity_threshold:
                            results.append(AudioSearchResult(
                                entry=entry,
                                similarity=sim,
                                match_type="speaker",
                            ))
                            break  # One match per entry

        results.sort(key=lambda r: r.similarity, reverse=True)

        self._stats["searches_performed"] += 1

        return results[:limit]

    async def search_by_tags(
        self,
        tags: list[str],
        require_all: bool = False,
        limit: int = 10,
    ) -> list[AudioSearchResult]:
        """
        Search for audio by tags.

        Args:
            tags: Tags to search for
            require_all: If True, all tags must be present
            limit: Maximum results

        Returns:
            List of AudioSearchResult
        """
        if not tags:
            return []

        if require_all:
            matching_ids = None
            for tag in tags:
                tag_ids = self._tag_index.get(tag, set())
                if matching_ids is None:
                    matching_ids = tag_ids.copy()
                else:
                    matching_ids &= tag_ids
            matching_ids = matching_ids or set()
        else:
            matching_ids = set()
            for tag in tags:
                matching_ids |= self._tag_index.get(tag, set())

        results = []
        for entry_id in matching_ids:
            if entry_id in self._entries:
                entry = self._entries[entry_id]
                # Score by tag overlap
                overlap = len(set(tags) & set(entry.tags))
                similarity = overlap / len(tags)
                results.append(AudioSearchResult(
                    entry=entry,
                    similarity=similarity,
                    match_type="tag",
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)

        self._stats["searches_performed"] += 1

        return results[:limit]

    async def get(self, entry_id: str) -> Optional[AudioMemoryEntry]:
        """Get a specific memory entry."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._stats["entries_retrieved"] += 1
        return entry

    async def update_importance(self, entry_id: str, importance: float) -> bool:
        """Update the importance of a memory entry."""
        if entry_id in self._entries:
            self._entries[entry_id].importance = max(0, min(1, importance))
            return True
        return False

    async def add_tags(self, entry_id: str, tags: list[str]) -> bool:
        """Add tags to a memory entry."""
        if entry_id not in self._entries:
            return False

        entry = self._entries[entry_id]
        for tag in tags:
            if tag not in entry.tags:
                entry.tags.append(tag)
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(entry_id)

        return True

    # ========================
    # Speaker Registration
    # ========================

    async def register_speaker(
        self,
        speaker: Speaker,
        name: Optional[str] = None,
    ) -> Speaker:
        """
        Register a speaker for identification.

        Args:
            speaker: Speaker with embedding
            name: Optional speaker name

        Returns:
            Registered speaker
        """
        if name:
            speaker.name = name

        self._registered_speakers[speaker.id] = speaker
        logger.info("Speaker registered", speaker_id=speaker.id, name=name)

        return speaker

    async def identify_speaker(
        self,
        embedding: np.ndarray,
        threshold: float = 0.75,
    ) -> tuple[Optional[Speaker], float]:
        """
        Identify a speaker from registered speakers.

        Args:
            embedding: Speaker embedding
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (matched speaker or None, similarity score)
        """
        best_match = None
        best_similarity = 0.0

        for speaker in self._registered_speakers.values():
            if speaker.embedding is not None:
                sim = float(np.dot(embedding, speaker.embedding) /
                           (np.linalg.norm(embedding) * np.linalg.norm(speaker.embedding) + 1e-10))
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = speaker

        if best_similarity >= threshold:
            return best_match, best_similarity

        return None, best_similarity

    async def get_registered_speakers(self) -> list[Speaker]:
        """Get all registered speakers."""
        return list(self._registered_speakers.values())

    # ========================
    # Conversation Tracking
    # ========================

    async def start_conversation(
        self,
        title: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Conversation:
        """
        Start tracking a new conversation.

        Args:
            title: Optional conversation title
            metadata: Additional metadata

        Returns:
            New Conversation object
        """
        conversation = Conversation(
            title=title,
            metadata=metadata or {},
        )

        self._conversations[conversation.id] = conversation
        self._active_conversation = conversation.id
        self._stats["conversations_tracked"] += 1

        logger.info("Conversation started", conversation_id=conversation.id)
        return conversation

    async def add_to_conversation(
        self,
        memory_id: str,
        speaker_id: Optional[str] = None,
        text: str = "",
        start_time: float = 0.0,
        end_time: float = 0.0,
        conversation_id: Optional[str] = None,
    ) -> Optional[Conversation]:
        """
        Add a segment to an active conversation.

        Args:
            memory_id: ID of the stored memory
            speaker_id: Speaker ID if known
            text: Transcript text
            start_time: Start time in seconds
            end_time: End time in seconds
            conversation_id: Specific conversation (uses active if None)

        Returns:
            Updated Conversation or None
        """
        conv_id = conversation_id or self._active_conversation
        if not conv_id or conv_id not in self._conversations:
            return None

        conversation = self._conversations[conv_id]
        segment = ConversationSegment(
            memory_id=memory_id,
            speaker_id=speaker_id,
            text=text,
            start_time=start_time,
            end_time=end_time,
        )

        conversation.segments.append(segment)

        # Track speakers
        if speaker_id:
            # Try to get speaker name from registered speakers or memory
            if speaker_id in self._registered_speakers:
                speaker_name = self._registered_speakers[speaker_id].name or f"Speaker {speaker_id[:8]}"
            else:
                speaker_name = f"Speaker {len(conversation.speakers) + 1}"
            conversation.speakers[speaker_id] = speaker_name

        return conversation

    async def end_conversation(
        self,
        conversation_id: Optional[str] = None,
        topics: Optional[list[str]] = None,
    ) -> Optional[Conversation]:
        """
        End and finalize a conversation.

        Args:
            conversation_id: Specific conversation (uses active if None)
            topics: Detected topics

        Returns:
            Finalized Conversation or None
        """
        conv_id = conversation_id or self._active_conversation
        if not conv_id or conv_id not in self._conversations:
            return None

        conversation = self._conversations[conv_id]
        conversation.end_time = datetime.now()

        if topics:
            conversation.topics = topics

        if self._active_conversation == conv_id:
            self._active_conversation = None

        logger.info("Conversation ended",
                   conversation_id=conversation.id,
                   segments=len(conversation.segments))

        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    async def search_conversations(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Conversation]:
        """
        Search conversations by transcript content.

        Args:
            query: Text query
            limit: Maximum results

        Returns:
            List of matching Conversations
        """
        query_words = set(query.lower().split())
        results = []

        for conversation in self._conversations.values():
            transcript = conversation.full_transcript.lower()
            match_count = sum(1 for word in query_words if word in transcript)
            if match_count > 0:
                results.append((conversation, match_count))

        results.sort(key=lambda x: x[1], reverse=True)
        return [conv for conv, _ in results[:limit]]

    def count(self) -> int:
        """Get number of stored memories."""
        return len(self._entries)

    def get_all_tags(self) -> list[str]:
        """Get all unique tags."""
        return list(self._tag_index.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            **self._stats,
            "total_entries": len(self._entries),
            "total_conversations": len(self._conversations),
            "registered_speakers": len(self._registered_speakers),
            "unique_tags": len(self._tag_index),
            "faiss_available": self._faiss_index is not None,
        }
