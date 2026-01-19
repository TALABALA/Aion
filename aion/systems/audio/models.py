"""
AION Audio Models

Dataclasses for the Auditory Cortex system:
- AudioSegment: Raw audio data with metadata
- Transcript: Speech transcription with timing
- Speaker: Identified speaker profile
- AudioEvent: Detected sound events
- AudioScene: Complete audio scene understanding
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np


@dataclass
class TimeRange:
    """A time range in audio (seconds)."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return self.start < other.end and other.start < self.end

    def intersection(self, other: "TimeRange") -> Optional["TimeRange"]:
        """Get the intersection with another range."""
        if not self.overlaps(other):
            return None
        return TimeRange(
            start=max(self.start, other.start),
            end=min(self.end, other.end),
        )

    def contains(self, time: float) -> bool:
        """Check if a time point is within this range."""
        return self.start <= time <= self.end

    def to_dict(self) -> dict[str, float]:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
        }


@dataclass
class FrequencyRange:
    """A frequency range in audio (Hz)."""
    low: float
    high: float

    @property
    def bandwidth(self) -> float:
        return self.high - self.low

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2

    def to_dict(self) -> dict[str, float]:
        return {
            "low": self.low,
            "high": self.high,
            "bandwidth": self.bandwidth,
            "center": self.center,
        }


@dataclass
class AudioSegment:
    """
    A segment of audio with metadata.

    Represents a portion of audio with optional raw data,
    spectral features, and learned embeddings.
    """
    id: str
    start_time: float  # seconds
    end_time: float  # seconds
    sample_rate: int
    channels: int
    waveform: Optional[np.ndarray] = None  # Raw audio data (samples,) or (channels, samples)
    spectrogram: Optional[np.ndarray] = None  # Mel spectrogram or similar
    embeddings: Optional[np.ndarray] = None  # CLAP or similar audio embeddings
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time

    @property
    def time_range(self) -> TimeRange:
        """Get as TimeRange object."""
        return TimeRange(self.start_time, self.end_time)

    @property
    def num_samples(self) -> Optional[int]:
        """Number of samples if waveform is available."""
        if self.waveform is not None:
            if self.waveform.ndim == 1:
                return len(self.waveform)
            return self.waveform.shape[-1]
        return int(self.duration * self.sample_rate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "num_samples": self.num_samples,
            "has_waveform": self.waveform is not None,
            "has_spectrogram": self.spectrogram is not None,
            "has_embeddings": self.embeddings is not None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_array(
        cls,
        waveform: np.ndarray,
        sample_rate: int,
        start_time: float = 0.0,
        segment_id: Optional[str] = None,
        **metadata,
    ) -> "AudioSegment":
        """Create AudioSegment from numpy array."""
        if waveform.ndim == 1:
            channels = 1
            duration = len(waveform) / sample_rate
        else:
            channels = waveform.shape[0]
            duration = waveform.shape[1] / sample_rate

        return cls(
            id=segment_id or str(uuid.uuid4()),
            start_time=start_time,
            end_time=start_time + duration,
            sample_rate=sample_rate,
            channels=channels,
            waveform=waveform,
            metadata=metadata,
        )


@dataclass
class Word:
    """A single word in a transcription."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
        }


@dataclass
class TranscriptSegment:
    """
    A segment of transcription with timing.

    Represents a phrase, sentence, or utterance within a full transcript.
    """
    id: str
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None
    words: list[Word] = field(default_factory=list)
    language: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def word_count(self) -> int:
        return len(self.words) if self.words else len(self.text.split())

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start_time, self.end_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
            "words": [w.to_dict() for w in self.words],
            "word_count": self.word_count,
            "language": self.language,
        }


@dataclass
class Speaker:
    """
    Identified speaker in audio.

    Represents a unique speaker with their voice embedding
    and all segments where they speak.
    """
    id: str
    name: Optional[str] = None
    embedding: Optional[np.ndarray] = None  # Speaker voice embedding
    segments: list[TimeRange] = field(default_factory=list)  # Time ranges where speaker talks
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_speaking_time(self) -> float:
        """Total time this speaker was speaking."""
        return sum(seg.duration for seg in self.segments)

    @property
    def segment_count(self) -> int:
        """Number of speaking segments."""
        return len(self.segments)

    def add_segment(self, start: float, end: float) -> None:
        """Add a speaking segment."""
        self.segments.append(TimeRange(start, end))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "has_embedding": self.embedding is not None,
            "segments": [s.to_dict() for s in self.segments],
            "segment_count": self.segment_count,
            "total_speaking_time": self.total_speaking_time,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def similarity(self, other: "Speaker") -> float:
        """Compute cosine similarity with another speaker."""
        if self.embedding is None or other.embedding is None:
            return 0.0

        dot = np.dot(self.embedding, other.embedding)
        norm1 = np.linalg.norm(self.embedding)
        norm2 = np.linalg.norm(other.embedding)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


@dataclass
class Transcript:
    """
    Speech transcription with rich metadata.

    Full transcription of audio including:
    - Complete text
    - Segment-level timing
    - Speaker diarization
    - Language detection
    """
    id: str
    text: str
    language: str
    confidence: float
    segments: list[TranscriptSegment]
    speakers: list[Speaker]
    audio_id: str
    duration: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Total word count."""
        return sum(seg.word_count for seg in self.segments)

    @property
    def speaker_count(self) -> int:
        """Number of unique speakers."""
        return len(self.speakers)

    def get_text_for_speaker(self, speaker_id: str) -> str:
        """Get all text spoken by a specific speaker."""
        return " ".join(
            seg.text for seg in self.segments
            if seg.speaker_id == speaker_id
        )

    def get_segments_in_range(
        self,
        start: float,
        end: float,
    ) -> list[TranscriptSegment]:
        """Get all segments within a time range."""
        time_range = TimeRange(start, end)
        return [
            seg for seg in self.segments
            if seg.time_range.overlaps(time_range)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "segments": [s.to_dict() for s in self.segments],
            "speakers": [s.to_dict() for s in self.speakers],
            "audio_id": self.audio_id,
            "duration": self.duration,
            "word_count": self.word_count,
            "speaker_count": self.speaker_count,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_srt(self) -> str:
        """Export as SRT subtitle format."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start_ts = self._format_srt_time(seg.start_time)
            end_ts = self._format_srt_time(seg.end_time)
            speaker_prefix = f"[{seg.speaker_id}] " if seg.speaker_id else ""
            lines.extend([
                str(i),
                f"{start_ts} --> {end_ts}",
                f"{speaker_prefix}{seg.text}",
                "",
            ])
        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time for SRT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@dataclass
class AudioEvent:
    """
    A detected audio event.

    Represents a specific sound event like speech, music,
    environmental sounds (dog bark, door knock, etc.).
    """
    id: str
    label: str  # e.g., "speech", "music", "dog_bark", "door_knock"
    category: str  # e.g., "speech", "music", "ambient", "action"
    start_time: float
    end_time: float
    confidence: float
    frequency_range: Optional[FrequencyRange] = None
    source_location: Optional[tuple[float, float, float]] = None  # 3D position (x, y, z)
    intensity_db: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start_time, self.end_time)

    def overlaps(self, other: "AudioEvent") -> bool:
        """Check if this event overlaps temporally with another."""
        return self.time_range.overlaps(other.time_range)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "frequency_range": self.frequency_range.to_dict() if self.frequency_range else None,
            "source_location": self.source_location,
            "intensity_db": self.intensity_db,
            "metadata": self.metadata,
        }


@dataclass
class MusicAnalysis:
    """
    Music analysis results.

    Contains tempo, key, mode, and other musical features.
    """
    tempo_bpm: float
    key: str  # e.g., "C", "F#", "Bb"
    mode: str  # "major" or "minor"
    time_signature: str  # e.g., "4/4", "3/4", "6/8"
    genre: Optional[str] = None
    mood: Optional[str] = None  # e.g., "happy", "sad", "energetic"
    energy: float = 0.5  # 0-1 scale
    danceability: float = 0.5  # 0-1 scale
    instrumentalness: float = 0.5  # 0-1 scale
    valence: float = 0.5  # Musical positiveness 0-1
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo_bpm": self.tempo_bpm,
            "key": self.key,
            "mode": self.mode,
            "time_signature": self.time_signature,
            "genre": self.genre,
            "mood": self.mood,
            "energy": self.energy,
            "danceability": self.danceability,
            "instrumentalness": self.instrumentalness,
            "valence": self.valence,
            "confidence": self.confidence,
        }


@dataclass
class AudioRelation:
    """A temporal or causal relation between audio events."""
    source_id: str
    target_id: str
    relation_type: str  # "before", "after", "during", "causes", "responds_to"
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
        }


@dataclass
class AudioScene:
    """
    Complete audio scene understanding.

    Comprehensive analysis of an audio segment including:
    - Detected events (speech, music, sounds)
    - Identified speakers
    - Transcription
    - Ambient description
    - Music analysis
    - Emotional tone
    """
    id: str
    audio_id: str
    duration: float
    events: list[AudioEvent]
    speakers: list[Speaker]
    relations: list[AudioRelation] = field(default_factory=list)
    transcript: Optional[Transcript] = None
    ambient_description: str = ""  # "quiet office", "busy street", etc.
    ambient_category: str = ""  # "indoor", "outdoor", "vehicle", etc.
    music_analysis: Optional[MusicAnalysis] = None
    emotional_tone: Optional[str] = None  # Overall emotional content
    noise_level_db: float = 0.0
    signal_to_noise_ratio: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def has_speech(self) -> bool:
        return any(e.category == "speech" for e in self.events)

    @property
    def has_music(self) -> bool:
        return any(e.category == "music" for e in self.events)

    def get_events_by_category(self, category: str) -> list[AudioEvent]:
        """Get all events of a specific category."""
        return [e for e in self.events if e.category == category]

    def get_events_at_time(self, time: float) -> list[AudioEvent]:
        """Get all events occurring at a specific time."""
        return [e for e in self.events if e.time_range.contains(time)]

    def describe(self) -> str:
        """Generate a natural language description of the scene."""
        parts = []

        if self.ambient_description:
            parts.append(f"The audio is from {self.ambient_description}.")

        if self.has_speech and self.transcript:
            speaker_count = len(self.speakers)
            if speaker_count == 1:
                parts.append("One person is speaking.")
            else:
                parts.append(f"{speaker_count} people are conversing.")

        if self.has_music and self.music_analysis:
            music = self.music_analysis
            parts.append(
                f"There is {music.mood or 'background'} music "
                f"in {music.key} {music.mode} at {music.tempo_bpm:.0f} BPM."
            )

        # Describe notable sound events
        sound_events = [e for e in self.events if e.category not in ("speech", "music")]
        if sound_events:
            labels = list(set(e.label for e in sound_events[:5]))
            parts.append(f"Notable sounds include: {', '.join(labels)}.")

        if self.emotional_tone:
            parts.append(f"The overall tone is {self.emotional_tone}.")

        return " ".join(parts) if parts else "Audio content detected."

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "audio_id": self.audio_id,
            "duration": self.duration,
            "events": [e.to_dict() for e in self.events],
            "event_count": self.event_count,
            "speakers": [s.to_dict() for s in self.speakers],
            "relations": [r.to_dict() for r in self.relations],
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "ambient_description": self.ambient_description,
            "ambient_category": self.ambient_category,
            "music_analysis": self.music_analysis.to_dict() if self.music_analysis else None,
            "emotional_tone": self.emotional_tone,
            "noise_level_db": self.noise_level_db,
            "signal_to_noise_ratio": self.signal_to_noise_ratio,
            "has_speech": self.has_speech,
            "has_music": self.has_music,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AudioMemoryEntry:
    """An entry in audio memory."""
    id: str
    audio_hash: str
    embedding: np.ndarray
    scene: AudioScene
    transcript_text: Optional[str]
    duration: float
    metadata: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "audio_hash": self.audio_hash,
            "scene": self.scene.to_dict(),
            "transcript_text": self.transcript_text,
            "duration": self.duration,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "importance": self.importance,
        }


@dataclass
class AudioSearchResult:
    """Result of an audio memory search."""
    entry: AudioMemoryEntry
    similarity: float
    match_type: str = "embedding"  # "embedding", "speaker", "transcript"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "similarity": self.similarity,
            "match_type": self.match_type,
        }


@dataclass
class AudioAnalysisResult:
    """Complete result of audio analysis."""
    audio_segment: AudioSegment
    scene: AudioScene
    transcript: Optional[Transcript]
    attention_regions: list[TimeRange]
    reasoning: Optional[str] = None
    similar_memories: list[AudioMemoryEntry] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_segment": self.audio_segment.to_dict(),
            "scene": self.scene.to_dict(),
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "attention_regions": [r.to_dict() for r in self.attention_regions],
            "reasoning": self.reasoning,
            "similar_memories_count": len(self.similar_memories),
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class VoiceProfile:
    """A stored voice profile for speaker identification."""
    id: str
    name: str
    embedding: np.ndarray
    sample_count: int = 1
    total_duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_embedding(
        self,
        new_embedding: np.ndarray,
        duration: float,
    ) -> None:
        """Update embedding with a new sample (running average)."""
        # Weighted average based on sample count
        weight = self.sample_count / (self.sample_count + 1)
        self.embedding = weight * self.embedding + (1 - weight) * new_embedding
        self.sample_count += 1
        self.total_duration += duration
        self.updated_at = datetime.now()

    def similarity(self, other_embedding: np.ndarray) -> float:
        """Compute cosine similarity with another embedding."""
        dot = np.dot(self.embedding, other_embedding)
        norm1 = np.linalg.norm(self.embedding)
        norm2 = np.linalg.norm(other_embedding)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "sample_count": self.sample_count,
            "total_duration": self.total_duration,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }
