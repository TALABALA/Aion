"""
AION Audio Models

Comprehensive data structures for the Auditory Cortex system:
- Audio segments and waveforms
- Speech transcription with diarization
- Speaker identification and verification
- Audio events and scene understanding
- Music analysis and understanding
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import numpy as np


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    WEBM = "webm"


class AudioEventType(Enum):
    """Types of audio events."""
    SPEECH = "speech"
    MUSIC = "music"
    SILENCE = "silence"
    NOISE = "noise"
    ENVIRONMENTAL = "environmental"
    MECHANICAL = "mechanical"
    ANIMAL = "animal"
    ALERT = "alert"
    UNKNOWN = "unknown"


class EmotionalTone(Enum):
    """Emotional tones detected in audio."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    CALM = "calm"
    EXCITED = "excited"
    TENSE = "tense"


class SpeechStyle(Enum):
    """Speech delivery styles."""
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    NARRATIVE = "narrative"
    EMOTIONAL = "emotional"
    WHISPERED = "whispered"
    SHOUTED = "shouted"


@dataclass
class TimeRange:
    """A time range in audio."""
    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        """Duration of the time range."""
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        """Midpoint of the time range."""
        return (self.start + self.end) / 2

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return self.start < other.end and other.start < self.end

    def overlap_duration(self, other: "TimeRange") -> float:
        """Calculate overlap duration with another range."""
        if not self.overlaps(other):
            return 0.0
        return min(self.end, other.end) - max(self.start, other.start)

    def contains(self, time: float) -> bool:
        """Check if a time point is within this range."""
        return self.start <= time <= self.end

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "duration": self.duration}


@dataclass
class AudioSegment:
    """
    A segment of audio with metadata.

    Core data structure for representing audio data throughout the system.
    Can contain raw waveform, spectrogram, and embeddings.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = 0.0  # seconds
    end_time: float = 0.0  # seconds
    sample_rate: int = 16000
    channels: int = 1
    waveform: Optional[np.ndarray] = None  # Raw audio data (time,) or (channels, time)
    spectrogram: Optional[np.ndarray] = None  # Mel spectrogram
    embeddings: Optional[np.ndarray] = None  # CLAP or similar embeddings
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self.waveform is not None:
            if self.waveform.ndim == 1:
                return len(self.waveform) / self.sample_rate
            return self.waveform.shape[-1] / self.sample_rate
        return self.end_time - self.start_time

    @property
    def num_samples(self) -> int:
        """Number of audio samples."""
        if self.waveform is not None:
            if self.waveform.ndim == 1:
                return len(self.waveform)
            return self.waveform.shape[-1]
        return int(self.duration * self.sample_rate)

    @property
    def time_range(self) -> TimeRange:
        """Get as TimeRange."""
        return TimeRange(self.start_time, self.end_time if self.end_time > 0 else self.duration)

    def get_subsegment(self, start: float, end: float) -> "AudioSegment":
        """Extract a subsegment of audio."""
        if self.waveform is None:
            return AudioSegment(
                start_time=start,
                end_time=end,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )

        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)

        if self.waveform.ndim == 1:
            sub_waveform = self.waveform[start_sample:end_sample]
        else:
            sub_waveform = self.waveform[:, start_sample:end_sample]

        return AudioSegment(
            start_time=start,
            end_time=end,
            sample_rate=self.sample_rate,
            channels=self.channels,
            waveform=sub_waveform,
        )

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
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Word:
    """A single word with timing and confidence."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
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

    Represents a phrase or sentence with word-level timestamps
    and optional speaker attribution.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0
    speaker_id: Optional[str] = None
    words: list[Word] = field(default_factory=list)
    language: Optional[str] = None
    emotional_tone: Optional[EmotionalTone] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start_time, self.end_time)

    @property
    def word_count(self) -> int:
        return len(self.words) if self.words else len(self.text.split())

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
            "language": self.language,
            "emotional_tone": self.emotional_tone.value if self.emotional_tone else None,
        }


@dataclass
class Speaker:
    """
    Identified speaker in audio.

    Contains speaker metadata, voice embedding, and
    all segments attributed to this speaker.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    embedding: Optional[np.ndarray] = None  # Voice embedding for identification
    segments: list[TimeRange] = field(default_factory=list)  # When they spoke
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Voice characteristics
    gender: Optional[str] = None  # "male", "female", "other"
    age_group: Optional[str] = None  # "child", "young_adult", "adult", "senior"
    voice_quality: Optional[str] = None  # "clear", "hoarse", "nasal", etc.

    @property
    def total_speaking_time(self) -> float:
        """Total time this speaker spoke."""
        return sum(seg.duration for seg in self.segments)

    @property
    def segment_count(self) -> int:
        """Number of speaking segments."""
        return len(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "segments": [s.to_dict() for s in self.segments],
            "confidence": self.confidence,
            "total_speaking_time": self.total_speaking_time,
            "segment_count": self.segment_count,
            "gender": self.gender,
            "age_group": self.age_group,
            "voice_quality": self.voice_quality,
            "metadata": self.metadata,
        }


@dataclass
class Transcript:
    """
    Speech transcription with rich metadata.

    Complete transcription result including word-level timestamps,
    speaker diarization, and language detection.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""  # Full transcription text
    language: str = "en"
    language_probability: float = 1.0
    confidence: float = 1.0
    segments: list[TranscriptSegment] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)
    audio_id: Optional[str] = None
    duration: float = 0.0
    word_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.text and self.segments:
            self.text = " ".join(seg.text for seg in self.segments)
        if not self.word_count:
            self.word_count = len(self.text.split())

    @property
    def speaker_count(self) -> int:
        """Number of unique speakers."""
        return len(self.speakers)

    def get_speaker_segments(self, speaker_id: str) -> list[TranscriptSegment]:
        """Get all segments for a specific speaker."""
        return [seg for seg in self.segments if seg.speaker_id == speaker_id]

    def get_text_by_speaker(self) -> dict[str, str]:
        """Get transcription text grouped by speaker."""
        result: dict[str, str] = {}
        for seg in self.segments:
            speaker = seg.speaker_id or "unknown"
            if speaker not in result:
                result[speaker] = ""
            result[speaker] += seg.text + " "
        return {k: v.strip() for k, v in result.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "confidence": self.confidence,
            "segments": [s.to_dict() for s in self.segments],
            "speakers": [s.to_dict() for s in self.speakers],
            "audio_id": self.audio_id,
            "duration": self.duration,
            "word_count": self.word_count,
            "speaker_count": self.speaker_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AudioEvent:
    """
    A detected audio event.

    Represents discrete events like speech, music, environmental sounds,
    with temporal boundaries and optional spatial information.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""  # e.g., "door_knock", "speech", "music", "dog_bark"
    event_type: AudioEventType = AudioEventType.UNKNOWN
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    source_location: Optional[tuple[float, float, float]] = None  # 3D position (x, y, z)
    intensity: float = 0.0  # Relative loudness 0-1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start_time, self.end_time)

    def overlaps(self, other: "AudioEvent") -> bool:
        """Check if this event overlaps with another."""
        return self.time_range.overlaps(other.time_range)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "event_type": self.event_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "source_location": self.source_location,
            "intensity": self.intensity,
            "metadata": self.metadata,
        }


@dataclass
class MusicAnalysis:
    """
    Music analysis results.

    Contains tempo, key, mood, genre, and structural analysis.
    """
    tempo: float = 0.0  # BPM
    tempo_confidence: float = 0.0
    key: Optional[str] = None  # e.g., "C major", "A minor"
    key_confidence: float = 0.0
    time_signature: str = "4/4"
    mode: str = "major"  # "major" or "minor"

    # Energy and dynamics
    energy: float = 0.0  # 0-1 scale
    danceability: float = 0.0  # 0-1 scale
    valence: float = 0.5  # 0 (sad) to 1 (happy)

    # Genre and mood
    genre: Optional[str] = None
    genre_confidence: float = 0.0
    mood: Optional[str] = None  # "energetic", "calm", "melancholic", etc.

    # Structure
    beats: list[float] = field(default_factory=list)  # Beat timestamps
    downbeats: list[float] = field(default_factory=list)  # Measure starts
    sections: list[dict[str, Any]] = field(default_factory=list)  # Verse, chorus, etc.

    # Instrumentation
    instruments_detected: list[str] = field(default_factory=list)
    has_vocals: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo": self.tempo,
            "tempo_confidence": self.tempo_confidence,
            "key": self.key,
            "key_confidence": self.key_confidence,
            "time_signature": self.time_signature,
            "mode": self.mode,
            "energy": self.energy,
            "danceability": self.danceability,
            "valence": self.valence,
            "genre": self.genre,
            "genre_confidence": self.genre_confidence,
            "mood": self.mood,
            "beats": self.beats[:20] if self.beats else [],  # Limit for serialization
            "beat_count": len(self.beats),
            "sections": self.sections,
            "instruments_detected": self.instruments_detected,
            "has_vocals": self.has_vocals,
            "metadata": self.metadata,
        }


@dataclass
class AudioQuality:
    """Audio quality metrics."""
    sample_rate: int = 0
    bit_depth: int = 16
    channels: int = 1
    codec: Optional[str] = None
    bitrate: Optional[int] = None  # kbps

    # Quality metrics
    signal_to_noise_ratio: float = 0.0  # dB
    peak_amplitude: float = 0.0
    rms_amplitude: float = 0.0
    dynamic_range: float = 0.0  # dB
    clipping_detected: bool = False
    noise_level: float = 0.0  # 0-1 scale

    # Issues
    has_distortion: bool = False
    has_background_noise: bool = False
    has_echo: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "signal_to_noise_ratio": self.signal_to_noise_ratio,
            "peak_amplitude": self.peak_amplitude,
            "rms_amplitude": self.rms_amplitude,
            "dynamic_range": self.dynamic_range,
            "clipping_detected": self.clipping_detected,
            "noise_level": self.noise_level,
            "has_distortion": self.has_distortion,
            "has_background_noise": self.has_background_noise,
            "has_echo": self.has_echo,
        }


@dataclass
class AudioScene:
    """
    Complete audio scene understanding.

    Combines all analysis results for comprehensive
    audio scene representation.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_id: Optional[str] = None
    duration: float = 0.0

    # Core analysis
    events: list[AudioEvent] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)
    transcript: Optional[Transcript] = None

    # Scene classification
    scene_type: str = "unknown"  # "conversation", "music", "nature", "urban", etc.
    ambient_description: str = ""  # "quiet office", "busy street", etc.
    location_type: Optional[str] = None  # "indoor", "outdoor", "vehicle"

    # Music analysis (if music present)
    music_analysis: Optional[MusicAnalysis] = None
    has_music: bool = False
    music_is_foreground: bool = False

    # Emotional analysis
    emotional_tone: Optional[EmotionalTone] = None
    emotional_confidence: float = 0.0

    # Audio quality
    quality: Optional[AudioQuality] = None
    noise_level_db: float = 0.0
    average_loudness_db: float = 0.0

    # Embeddings for retrieval
    scene_embedding: Optional[np.ndarray] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def speaker_count(self) -> int:
        return len(self.speakers)

    @property
    def has_speech(self) -> bool:
        return any(e.event_type == AudioEventType.SPEECH for e in self.events)

    def get_events_by_type(self, event_type: AudioEventType) -> list[AudioEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_in_range(self, time_range: TimeRange) -> list[AudioEvent]:
        """Get events within a time range."""
        return [e for e in self.events if e.time_range.overlaps(time_range)]

    def describe(self) -> str:
        """Generate a natural language description of the audio scene."""
        descriptions = []

        # Scene type
        descriptions.append(f"This is a {self.scene_type} audio scene")
        if self.ambient_description:
            descriptions.append(f"in a {self.ambient_description} environment")
        descriptions.append(f"lasting {self.duration:.1f} seconds.")

        # Speakers
        if self.speakers:
            if len(self.speakers) == 1:
                descriptions.append("There is 1 speaker.")
            else:
                descriptions.append(f"There are {len(self.speakers)} speakers.")

        # Events
        event_types = {}
        for event in self.events:
            event_types[event.label] = event_types.get(event.label, 0) + 1

        if event_types:
            event_strs = [f"{count} {label}{'s' if count > 1 else ''}"
                        for label, count in event_types.items()]
            descriptions.append(f"Detected events: {', '.join(event_strs)}.")

        # Music
        if self.has_music and self.music_analysis:
            descriptions.append(
                f"Music detected at {self.music_analysis.tempo:.0f} BPM"
                f" in {self.music_analysis.key or 'unknown key'}."
            )

        # Emotional tone
        if self.emotional_tone:
            descriptions.append(f"The overall tone is {self.emotional_tone.value}.")

        return " ".join(descriptions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "audio_id": self.audio_id,
            "duration": self.duration,
            "events": [e.to_dict() for e in self.events],
            "event_count": self.event_count,
            "speakers": [s.to_dict() for s in self.speakers],
            "speaker_count": self.speaker_count,
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "scene_type": self.scene_type,
            "ambient_description": self.ambient_description,
            "location_type": self.location_type,
            "music_analysis": self.music_analysis.to_dict() if self.music_analysis else None,
            "has_music": self.has_music,
            "has_speech": self.has_speech,
            "music_is_foreground": self.music_is_foreground,
            "emotional_tone": self.emotional_tone.value if self.emotional_tone else None,
            "emotional_confidence": self.emotional_confidence,
            "quality": self.quality.to_dict() if self.quality else None,
            "noise_level_db": self.noise_level_db,
            "average_loudness_db": self.average_loudness_db,
            "description": self.describe(),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class VoiceCharacteristics:
    """Detailed voice characteristics for voice cloning and synthesis."""
    pitch_mean: float = 0.0  # Hz
    pitch_std: float = 0.0
    pitch_range: tuple[float, float] = (0.0, 0.0)
    speaking_rate: float = 0.0  # words per minute
    energy_mean: float = 0.0
    energy_std: float = 0.0

    # Prosodic features
    intonation_pattern: str = "neutral"  # "rising", "falling", "varied"
    rhythm_regularity: float = 0.5  # 0-1, how regular speech rhythm is
    pause_frequency: float = 0.0  # pauses per minute
    average_pause_duration: float = 0.0  # seconds

    # Voice quality
    breathiness: float = 0.0  # 0-1
    creakiness: float = 0.0  # 0-1
    nasality: float = 0.0  # 0-1

    # Spectral characteristics
    formant_frequencies: list[float] = field(default_factory=list)  # F1, F2, F3...
    spectral_tilt: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "pitch_range": self.pitch_range,
            "speaking_rate": self.speaking_rate,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "intonation_pattern": self.intonation_pattern,
            "rhythm_regularity": self.rhythm_regularity,
            "pause_frequency": self.pause_frequency,
            "average_pause_duration": self.average_pause_duration,
            "breathiness": self.breathiness,
            "creakiness": self.creakiness,
            "nasality": self.nasality,
            "formant_frequencies": self.formant_frequencies,
            "spectral_tilt": self.spectral_tilt,
        }


@dataclass
class SynthesisRequest:
    """Request for speech synthesis."""
    text: str
    voice_id: Optional[str] = None
    voice_embedding: Optional[np.ndarray] = None
    language: str = "en"
    speed: float = 1.0  # 0.5-2.0
    pitch: float = 1.0  # 0.5-2.0
    energy: float = 1.0  # 0.5-2.0
    style: SpeechStyle = SpeechStyle.CONVERSATIONAL
    emotion: Optional[EmotionalTone] = None

    # SSML-like controls
    emphasis_words: list[str] = field(default_factory=list)
    pause_after_sentences: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "voice_id": self.voice_id,
            "language": self.language,
            "speed": self.speed,
            "pitch": self.pitch,
            "energy": self.energy,
            "style": self.style.value,
            "emotion": self.emotion.value if self.emotion else None,
            "emphasis_words": self.emphasis_words,
            "pause_after_sentences": self.pause_after_sentences,
            "metadata": self.metadata,
        }


@dataclass
class SynthesisResult:
    """Result of speech synthesis."""
    audio: AudioSegment
    text: str
    voice_id: Optional[str] = None
    duration: float = 0.0
    processing_time_ms: float = 0.0

    # Alignment info
    word_timestamps: list[Word] = field(default_factory=list)
    phoneme_timestamps: list[dict[str, Any]] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio": self.audio.to_dict(),
            "text": self.text,
            "voice_id": self.voice_id,
            "duration": self.duration,
            "processing_time_ms": self.processing_time_ms,
            "word_timestamps": [w.to_dict() for w in self.word_timestamps],
            "metadata": self.metadata,
        }


@dataclass
class AudioMemoryEntry:
    """
    An entry in audio memory.

    Stores audio with embeddings, transcript, and metadata
    for later retrieval by similarity or content.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_hash: str = ""  # For deduplication
    embedding: Optional[np.ndarray] = None  # CLAP embedding for retrieval
    audio_scene: Optional[AudioScene] = None
    transcript: Optional[Transcript] = None
    context: str = ""  # User-provided context

    # Metadata
    source_path: Optional[str] = None
    duration: float = 0.0
    importance: float = 0.5  # 0-1, for memory prioritization

    # Access tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0

    # Tags for organization
    tags: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "audio_hash": self.audio_hash,
            "audio_scene": self.audio_scene.to_dict() if self.audio_scene else None,
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "context": self.context,
            "source_path": self.source_path,
            "duration": self.duration,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class AudioComparisonResult:
    """Result of comparing two audio samples."""
    audio1_id: str
    audio2_id: str

    # Similarity scores
    overall_similarity: float = 0.0  # 0-1
    acoustic_similarity: float = 0.0  # Based on embeddings
    content_similarity: float = 0.0  # Based on transcript
    speaker_similarity: float = 0.0  # If same speaker

    # Differences
    differences: list[str] = field(default_factory=list)

    # Common elements
    common_speakers: list[str] = field(default_factory=list)
    common_events: list[str] = field(default_factory=list)
    common_topics: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio1_id": self.audio1_id,
            "audio2_id": self.audio2_id,
            "overall_similarity": self.overall_similarity,
            "acoustic_similarity": self.acoustic_similarity,
            "content_similarity": self.content_similarity,
            "speaker_similarity": self.speaker_similarity,
            "differences": self.differences,
            "common_speakers": self.common_speakers,
            "common_events": self.common_events,
            "common_topics": self.common_topics,
            "metadata": self.metadata,
        }


@dataclass
class AudioReasoningResult:
    """Result of LLM-based audio reasoning."""
    question: str
    answer: str
    confidence: float = 0.0

    # Supporting evidence
    relevant_segments: list[TranscriptSegment] = field(default_factory=list)
    relevant_events: list[AudioEvent] = field(default_factory=list)

    # Reasoning trace
    reasoning_steps: list[str] = field(default_factory=list)

    # Source information
    audio_id: Optional[str] = None
    transcript_used: bool = False
    scene_analysis_used: bool = False

    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "relevant_segments": [s.to_dict() for s in self.relevant_segments],
            "relevant_events": [e.to_dict() for e in self.relevant_events],
            "reasoning_steps": self.reasoning_steps,
            "audio_id": self.audio_id,
            "transcript_used": self.transcript_used,
            "scene_analysis_used": self.scene_analysis_used,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }
