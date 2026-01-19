"""
AION Auditory Cortex

Unified audio processing system that integrates:
- Speech recognition with diarization
- Audio event detection and scene understanding
- Speaker identification and verification
- Text-to-speech generation
- Audio memory and retrieval
- Audio-based reasoning
"""

from __future__ import annotations

import asyncio
import io
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioSegment,
    AudioScene,
    AudioEvent,
    AudioRelation,
    Transcript,
    TranscriptSegment,
    Speaker,
    MusicAnalysis,
    TimeRange,
    AudioAnalysisResult,
    AudioMemoryEntry,
    VoiceProfile,
)
from aion.systems.audio.perception import AudioPerception
from aion.systems.audio.memory import AudioMemory

logger = structlog.get_logger(__name__)


@dataclass
class AudioAttention:
    """Attention focus on a temporal region of audio."""
    time_range: TimeRange
    focus_type: str  # "speech", "event", "music", "silence"
    target_id: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_range": self.time_range.to_dict(),
            "focus_type": self.focus_type,
            "target_id": self.target_id,
            "confidence": self.confidence,
        }


class AuditoryCortex:
    """
    AION Auditory Cortex - Complete audio perception system.

    The unified audio processing center that coordinates:
    - Low-level perception (transcription, event detection)
    - Mid-level understanding (scene graphs, speaker tracking)
    - High-level reasoning (audio QA, summarization)
    - Memory integration (similar audio retrieval)

    Provides:
    - Speech recognition with diarization
    - Audio event detection and scene understanding
    - Speaker identification and verification
    - Text-to-speech generation
    - Audio memory and retrieval
    """

    def __init__(
        self,
        enable_memory: bool = True,
        device: str = "auto",
        whisper_model: str = "openai/whisper-large-v3",
        speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        enable_diarization: bool = True,
        enable_tts: bool = True,
        max_audio_duration: float = 600.0,  # 10 minutes
        sample_rate: int = 16000,
    ):
        self.enable_memory = enable_memory
        self.device = device
        self.whisper_model = whisper_model
        self.speaker_model = speaker_model
        self.enable_diarization = enable_diarization
        self.enable_tts = enable_tts
        self.max_audio_duration = max_audio_duration
        self.sample_rate = sample_rate

        # Components
        self._perception = AudioPerception(
            whisper_model=whisper_model,
            speaker_model=speaker_model,
            device=device,
            sample_rate=sample_rate,
        )
        self._memory = AudioMemory() if enable_memory else None

        # TTS (lazy loaded)
        self._tts_model = None
        self._tts_available = False

        # Statistics
        self._stats = {
            "audio_processed": 0,
            "transcriptions": 0,
            "events_detected": 0,
            "questions_answered": 0,
            "speech_synthesized": 0,
            "total_processing_time_ms": 0.0,
            "total_audio_duration": 0.0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the auditory cortex."""
        if self._initialized:
            return

        logger.info("Initializing Auditory Cortex")

        await self._perception.initialize()

        if self._memory:
            await self._memory.initialize()

        if self.enable_tts:
            await self._initialize_tts()

        self._initialized = True
        logger.info(
            "Auditory Cortex initialized",
            device=self._perception.device,
            tts_available=self._tts_available,
        )

    async def _initialize_tts(self) -> None:
        """Initialize text-to-speech."""
        try:
            # Try to load a TTS model
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_tts)
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e}")

    def _load_tts(self) -> None:
        """Load TTS model (blocking)."""
        try:
            from TTS.api import TTS

            # Use a fast, high-quality model
            self._tts_model = TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False,
            )
            if self.device != "cpu":
                self._tts_model.to(self.device)

            self._tts_available = True
            logger.info("TTS model loaded successfully")

        except ImportError:
            logger.warning("Coqui TTS not available")
        except Exception as e:
            logger.warning(f"Failed to load TTS: {e}")

    async def shutdown(self) -> None:
        """Shutdown the auditory cortex."""
        await self._perception.shutdown()

        if self._memory:
            await self._memory.shutdown()

        self._tts_model = None
        self._initialized = False
        logger.info("Auditory Cortex shutdown")

    # ==================== Core Perception ====================

    async def process(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        query: Optional[str] = None,
        store_in_memory: bool = True,
    ) -> AudioAnalysisResult:
        """
        Process audio with full analysis.

        Args:
            audio: Audio source (path, bytes, or numpy array)
            query: Optional question about the audio
            store_in_memory: Whether to store in memory

        Returns:
            AudioAnalysisResult with complete analysis
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.monotonic()

        # Load audio
        waveform, sr = await self._perception.load_audio(audio)
        duration = len(waveform) / sr

        # Check duration limit
        if duration > self.max_audio_duration:
            logger.warning(
                "Audio exceeds maximum duration",
                duration=duration,
                max=self.max_audio_duration,
            )
            waveform = waveform[:int(self.max_audio_duration * sr)]
            duration = self.max_audio_duration

        # Create audio segment
        audio_hash = self._perception.compute_audio_hash(waveform)
        segment = AudioSegment.from_array(
            waveform, sr, segment_id=f"seg_{audio_hash[:8]}"
        )

        # Run analysis tasks in parallel
        transcribe_task = asyncio.create_task(
            self._perception.transcribe_with_diarization(waveform)
            if self.enable_diarization
            else self._perception.transcribe(waveform)
        )
        events_task = asyncio.create_task(
            self._perception.detect_events(waveform)
        )
        music_task = asyncio.create_task(
            self._perception.analyze_music(waveform)
        )
        embedding_task = asyncio.create_task(
            self._perception.get_audio_embedding(waveform)
        )

        transcript, events, music_analysis, embedding = await asyncio.gather(
            transcribe_task, events_task, music_task, embedding_task
        )

        # Build audio scene
        scene = await self._build_scene(
            waveform, sr, audio_hash, transcript, events, music_analysis
        )

        # Compute attention regions
        attention = self._compute_attention(scene, events)

        # Answer question if provided
        reasoning = None
        if query:
            reasoning = await self._answer_question_internal(
                waveform, scene, transcript, query
            )
            self._stats["questions_answered"] += 1

        # Store in memory
        if store_in_memory and self._memory:
            await self._memory.store(
                scene=scene,
                embedding=embedding,
                transcript_text=transcript.text if transcript else None,
                audio_hash=audio_hash,
                metadata={
                    "source": str(audio) if isinstance(audio, (str, Path)) else "array",
                    "query": query,
                },
            )

        # Search for similar memories
        similar_memories = []
        if self._memory and self._memory.count() > 0:
            results = await self._memory.search_by_embedding(embedding, limit=3)
            similar_memories = [r.entry for r in results]

        processing_time_ms = (time.monotonic() - start_time) * 1000

        # Update stats
        self._stats["audio_processed"] += 1
        self._stats["transcriptions"] += 1 if transcript else 0
        self._stats["events_detected"] += len(events)
        self._stats["total_processing_time_ms"] += processing_time_ms
        self._stats["total_audio_duration"] += duration

        return AudioAnalysisResult(
            audio_segment=segment,
            scene=scene,
            transcript=transcript,
            attention_regions=[a.time_range for a in attention],
            reasoning=reasoning,
            similar_memories=similar_memories,
            processing_time_ms=processing_time_ms,
            metadata={
                "source": str(audio) if isinstance(audio, (str, Path)) else "array",
                "query": query,
            },
        )

    async def _build_scene(
        self,
        waveform: np.ndarray,
        sr: int,
        audio_id: str,
        transcript: Optional[Transcript],
        events: list[AudioEvent],
        music_analysis: Optional[MusicAnalysis],
    ) -> AudioScene:
        """Build an audio scene from analysis results."""
        duration = len(waveform) / sr

        # Collect speakers from transcript
        speakers = transcript.speakers if transcript else []

        # Infer relations between events
        relations = self._infer_relations(events)

        # Estimate ambient description
        ambient_description = self._describe_ambient(events, music_analysis)

        # Estimate emotional tone
        emotional_tone = self._estimate_emotion(
            transcript, events, music_analysis
        )

        # Calculate noise level
        rms = np.sqrt(np.mean(waveform ** 2))
        noise_level_db = 20 * np.log10(rms + 1e-8)

        return AudioScene(
            id=f"scene_{uuid.uuid4().hex[:8]}",
            audio_id=audio_id,
            duration=duration,
            events=events,
            speakers=speakers,
            relations=relations,
            transcript=transcript,
            ambient_description=ambient_description,
            ambient_category=self._categorize_ambient(events),
            music_analysis=music_analysis,
            emotional_tone=emotional_tone,
            noise_level_db=noise_level_db,
        )

    def _infer_relations(
        self,
        events: list[AudioEvent],
    ) -> list[AudioRelation]:
        """Infer temporal relations between events."""
        relations = []

        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i >= j:
                    continue

                # Temporal relations
                if event1.end_time < event2.start_time:
                    relations.append(AudioRelation(
                        source_id=event1.id,
                        target_id=event2.id,
                        relation_type="before",
                        confidence=0.9,
                    ))
                elif event1.time_range.overlaps(event2.time_range):
                    relations.append(AudioRelation(
                        source_id=event1.id,
                        target_id=event2.id,
                        relation_type="during",
                        confidence=0.8,
                    ))

        return relations

    def _describe_ambient(
        self,
        events: list[AudioEvent],
        music: Optional[MusicAnalysis],
    ) -> str:
        """Generate ambient description."""
        descriptions = []

        # Check for common environments
        event_labels = {e.label.lower() for e in events}

        if "car" in event_labels or "engine" in event_labels:
            descriptions.append("vehicle interior")
        elif "crowd" in event_labels or "applause" in event_labels:
            descriptions.append("public gathering")
        elif "keyboard_typing" in event_labels:
            descriptions.append("office environment")
        elif "bird" in event_labels or "wind" in event_labels:
            descriptions.append("outdoor setting")
        elif "rain" in event_labels:
            descriptions.append("rainy weather")

        if music and music.confidence > 0.5:
            descriptions.append(f"{music.mood or 'background'} music")

        speech_events = [e for e in events if e.category == "speech"]
        if speech_events:
            descriptions.append("conversation")

        return ", ".join(descriptions) if descriptions else "ambient audio"

    def _categorize_ambient(self, events: list[AudioEvent]) -> str:
        """Categorize the ambient environment."""
        categories = [e.category for e in events]

        if "vehicle" in categories:
            return "vehicle"
        elif any("outdoor" in e.label.lower() for e in events):
            return "outdoor"
        elif any("office" in e.label.lower() for e in events):
            return "indoor"
        elif "music" in categories:
            return "music"
        else:
            return "indoor"  # Default

    def _estimate_emotion(
        self,
        transcript: Optional[Transcript],
        events: list[AudioEvent],
        music: Optional[MusicAnalysis],
    ) -> Optional[str]:
        """Estimate emotional tone."""
        emotions = []

        # From event labels
        event_labels = {e.label.lower() for e in events}
        if "laughter" in event_labels:
            emotions.append("joyful")
        if "crying" in event_labels:
            emotions.append("sad")
        if "applause" in event_labels:
            emotions.append("celebratory")

        # From music
        if music and music.mood:
            emotions.append(music.mood)

        return emotions[0] if emotions else None

    def _compute_attention(
        self,
        scene: AudioScene,
        events: list[AudioEvent],
    ) -> list[AudioAttention]:
        """Compute attention focus areas for audio."""
        attention = []

        # Add attention for high-confidence events
        for event in sorted(events, key=lambda e: e.confidence, reverse=True)[:5]:
            if event.confidence > 0.5:
                attention.append(AudioAttention(
                    time_range=event.time_range,
                    focus_type=event.category,
                    target_id=event.id,
                    confidence=event.confidence,
                ))

        # Add attention for speaker segments
        for speaker in scene.speakers[:3]:
            for segment in speaker.segments[:2]:
                attention.append(AudioAttention(
                    time_range=segment,
                    focus_type="speech",
                    target_id=speaker.id,
                    confidence=speaker.confidence,
                ))

        # Sort by time
        attention.sort(key=lambda a: a.time_range.start)

        return attention

    # ==================== Transcription ====================

    async def transcribe(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = None,
        enable_diarization: bool = True,
        enable_timestamps: bool = True,
    ) -> Transcript:
        """
        Transcribe speech to text with optional speaker diarization.

        Args:
            audio: Audio source
            language: Language code (auto-detect if None)
            enable_diarization: Enable speaker diarization
            enable_timestamps: Include word-level timestamps

        Returns:
            Transcript with segments and timing
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.monotonic()

        if enable_diarization and self.enable_diarization:
            transcript = await self._perception.transcribe_with_diarization(
                audio, language=language
            )
        else:
            transcript = await self._perception.transcribe(
                audio, language=language, return_timestamps=enable_timestamps
            )

        self._stats["transcriptions"] += 1
        self._stats["total_processing_time_ms"] += (
            (time.monotonic() - start_time) * 1000
        )

        return transcript

    # ==================== Event Detection ====================

    async def detect_events(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        threshold: float = 0.5,
    ) -> list[AudioEvent]:
        """
        Detect audio events (speech, music, environmental sounds).

        Args:
            audio: Audio source
            threshold: Detection confidence threshold

        Returns:
            List of detected AudioEvents
        """
        if not self._initialized:
            await self.initialize()

        events = await self._perception.detect_events(audio, threshold=threshold)
        self._stats["events_detected"] += len(events)

        return events

    async def understand_scene(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
    ) -> AudioScene:
        """
        Full audio scene understanding combining all capabilities.

        Args:
            audio: Audio source

        Returns:
            AudioScene with complete analysis
        """
        result = await self.process(audio, store_in_memory=False)
        return result.scene

    # ==================== Speaker Operations ====================

    async def identify_speaker(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        known_speakers: Optional[list[VoiceProfile]] = None,
    ) -> tuple[Optional[VoiceProfile], float]:
        """
        Identify speaker from audio.

        Args:
            audio: Audio source (should contain single speaker)
            known_speakers: List of known voice profiles

        Returns:
            Tuple of (matched_profile, similarity)
        """
        if not self._initialized:
            await self.initialize()

        # Use memory profiles if not provided
        if known_speakers is None and self._memory:
            known_speakers = self._memory.get_voice_profiles()

        if not known_speakers:
            return None, 0.0

        return await self._perception.identify_speaker(audio, known_speakers)

    async def register_speaker(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        name: str,
    ) -> VoiceProfile:
        """
        Register a new speaker from sample audio.

        Args:
            audio: Audio sample (should be single speaker)
            name: Speaker name

        Returns:
            Created VoiceProfile
        """
        if not self._initialized:
            await self.initialize()

        # Get speaker embedding
        embedding = await self._perception.get_speaker_embedding(audio)

        # Calculate duration
        waveform, sr = await self._perception.load_audio(audio)
        duration = len(waveform) / sr

        # Register in memory
        if self._memory:
            profile = await self._memory.register_voice_profile(
                name=name,
                embedding=embedding,
                duration=duration,
            )
        else:
            profile = VoiceProfile(
                id=str(uuid.uuid4()),
                name=name,
                embedding=embedding,
                total_duration=duration,
            )

        logger.info("Speaker registered", name=name, profile_id=profile.id)
        return profile

    async def verify_speaker(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        claimed_speaker: Union[VoiceProfile, Speaker],
        threshold: float = 0.75,
    ) -> tuple[bool, float]:
        """
        Verify if audio matches claimed speaker.

        Args:
            audio: Audio source
            claimed_speaker: Speaker to verify against
            threshold: Verification threshold

        Returns:
            Tuple of (is_verified, similarity)
        """
        if not self._initialized:
            await self.initialize()

        if isinstance(claimed_speaker, Speaker):
            if claimed_speaker.embedding is None:
                return False, 0.0
            profile = VoiceProfile(
                id=claimed_speaker.id,
                name=claimed_speaker.name or "",
                embedding=claimed_speaker.embedding,
            )
        else:
            profile = claimed_speaker

        return await self._perception.verify_speaker(audio, profile, threshold)

    # ==================== Speech Generation ====================

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ) -> AudioSegment:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice preset or speaker ID
            language: Language code
            speed: Speech speed multiplier

        Returns:
            AudioSegment with generated speech
        """
        if not self._initialized:
            await self.initialize()

        if not self._tts_available:
            logger.warning("TTS not available")
            return self._mock_tts_output(text)

        start_time = time.monotonic()

        loop = asyncio.get_event_loop()
        waveform, sr = await loop.run_in_executor(
            None,
            lambda: self._synthesize_sync(text, speed),
        )

        self._stats["speech_synthesized"] += 1
        self._stats["total_processing_time_ms"] += (
            (time.monotonic() - start_time) * 1000
        )

        return AudioSegment.from_array(
            waveform, sr,
            metadata={
                "text": text,
                "voice": voice,
                "language": language,
                "speed": speed,
            }
        )

    def _synthesize_sync(
        self,
        text: str,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Synchronous TTS."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            self._tts_model.tts_to_file(
                text=text,
                file_path=temp_path,
                speed=speed,
            )

            import soundfile as sf
            waveform, sr = sf.read(temp_path)

            return waveform.astype(np.float32), sr

        finally:
            import os
            os.unlink(temp_path)

    def _mock_tts_output(self, text: str) -> AudioSegment:
        """Return mock TTS output when unavailable."""
        # Generate silence with approximate duration
        words = len(text.split())
        duration = words * 0.5  # ~0.5 seconds per word

        waveform = np.zeros(int(duration * self.sample_rate), dtype=np.float32)

        return AudioSegment.from_array(
            waveform, self.sample_rate,
            metadata={"text": text, "mock": True}
        )

    async def clone_voice(
        self,
        reference_audio: Union[str, Path, bytes, np.ndarray],
        text: str,
    ) -> AudioSegment:
        """
        Generate speech in a cloned voice.

        Args:
            reference_audio: Reference audio for voice cloning
            text: Text to synthesize

        Returns:
            AudioSegment with cloned voice
        """
        if not self._initialized:
            await self.initialize()

        # Voice cloning requires specialized models
        # For now, fall back to regular synthesis
        logger.warning("Voice cloning not fully implemented, using default voice")

        return await self.synthesize(text)

    # ==================== Memory Operations ====================

    async def remember(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        context: Optional[str] = None,
        importance: float = 0.5,
    ) -> str:
        """
        Store audio in memory.

        Args:
            audio: Audio to store
            context: Optional context description
            importance: Importance score (0-1)

        Returns:
            Memory entry ID
        """
        if not self._memory:
            logger.warning("Memory not enabled")
            return ""

        result = await self.process(audio, store_in_memory=True)

        # Update importance if specified
        if importance != 0.5:
            entry = await self._memory.get(result.scene.id)
            if entry:
                entry.importance = importance

        return result.scene.id

    async def recall_similar(
        self,
        query: Union[str, np.ndarray],
        limit: int = 5,
    ) -> list[AudioMemoryEntry]:
        """
        Retrieve similar audio from memory.

        Args:
            query: Text query or audio embedding
            limit: Maximum results

        Returns:
            List of similar AudioMemoryEntries
        """
        if not self._memory:
            return []

        if isinstance(query, str):
            # Convert text query to embedding
            embedding = await self._perception.get_text_embedding(query)
        else:
            embedding = query

        results = await self._memory.search_by_embedding(embedding, limit=limit)
        return [r.entry for r in results]

    async def recall_by_speaker(
        self,
        speaker: Union[Speaker, VoiceProfile, str],
        limit: int = 10,
    ) -> list[AudioMemoryEntry]:
        """
        Retrieve audio by speaker.

        Args:
            speaker: Speaker to search for
            limit: Maximum results

        Returns:
            List of AudioMemoryEntries
        """
        if not self._memory:
            return []

        results = await self._memory.search_by_speaker(speaker, limit=limit)
        return [r.entry for r in results]

    async def recall_by_transcript(
        self,
        query: str,
        limit: int = 5,
    ) -> list[AudioMemoryEntry]:
        """
        Search memory by transcript content.

        Args:
            query: Text to search for
            limit: Maximum results

        Returns:
            List of AudioMemoryEntries
        """
        if not self._memory:
            return []

        results = await self._memory.search_by_text(query, limit=limit)
        return [r.entry for r in results]

    # ==================== Audio Reasoning ====================

    async def answer_question(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        question: str,
    ) -> str:
        """
        Answer a question about audio content.

        Args:
            audio: Audio source
            question: Question to answer

        Returns:
            Answer string
        """
        result = await self.process(audio, query=question, store_in_memory=False)
        return result.reasoning or "Unable to answer the question."

    async def _answer_question_internal(
        self,
        waveform: np.ndarray,
        scene: AudioScene,
        transcript: Optional[Transcript],
        question: str,
    ) -> str:
        """Internal question answering using scene context."""
        # Build context from scene
        context_parts = []

        if transcript and transcript.text:
            context_parts.append(f"Transcript: {transcript.text[:500]}")

        if scene.events:
            event_strs = [f"{e.label} ({e.confidence:.1%})" for e in scene.events[:10]]
            context_parts.append(f"Detected sounds: {', '.join(event_strs)}")

        if scene.speakers:
            context_parts.append(f"Speakers: {len(scene.speakers)}")

        if scene.music_analysis:
            music = scene.music_analysis
            context_parts.append(
                f"Music: {music.key} {music.mode}, {music.tempo_bpm:.0f} BPM, {music.mood or 'unknown mood'}"
            )

        if scene.ambient_description:
            context_parts.append(f"Environment: {scene.ambient_description}")

        context = "\n".join(context_parts)

        # Simple rule-based QA for common questions
        q_lower = question.lower()

        if "who" in q_lower and "speak" in q_lower:
            if scene.speakers:
                names = [s.name or f"Speaker {i+1}" for i, s in enumerate(scene.speakers)]
                return f"The speakers are: {', '.join(names)}"
            return "No speakers detected."

        if "what" in q_lower and ("sound" in q_lower or "hear" in q_lower):
            if scene.events:
                labels = list(set(e.label for e in scene.events))
                return f"I hear: {', '.join(labels[:5])}"
            return "No distinct sounds detected."

        if "music" in q_lower:
            if scene.music_analysis:
                m = scene.music_analysis
                return f"The music is in {m.key} {m.mode} at {m.tempo_bpm:.0f} BPM. The mood is {m.mood or 'neutral'}."
            return "No music detected."

        if "where" in q_lower or "environment" in q_lower or "location" in q_lower:
            return f"This sounds like {scene.ambient_description or 'an indoor environment'}."

        if "how many" in q_lower and "speaker" in q_lower:
            return f"There are {len(scene.speakers)} speakers."

        # Default: describe the audio
        return scene.describe()

    async def summarize(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
    ) -> str:
        """
        Generate a summary of audio content.

        Args:
            audio: Audio source

        Returns:
            Summary string
        """
        result = await self.process(audio, store_in_memory=False)
        return result.scene.describe()

    async def compare(
        self,
        audio1: Union[str, Path, bytes, np.ndarray],
        audio2: Union[str, Path, bytes, np.ndarray],
    ) -> dict[str, Any]:
        """
        Compare two audio samples.

        Args:
            audio1: First audio
            audio2: Second audio

        Returns:
            Comparison results
        """
        if not self._initialized:
            await self.initialize()

        # Analyze both
        result1 = await self.process(audio1, store_in_memory=False)
        result2 = await self.process(audio2, store_in_memory=False)

        # Compare embeddings
        emb1 = await self._perception.get_audio_embedding(audio1)
        emb2 = await self._perception.get_audio_embedding(audio2)

        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        similarity = float(np.dot(emb1_norm, emb2_norm))

        # Compare events
        events1 = {e.label for e in result1.scene.events}
        events2 = {e.label for e in result2.scene.events}

        common_events = events1 & events2
        unique_to_1 = events1 - events2
        unique_to_2 = events2 - events1

        # Compare speakers
        speakers1 = len(result1.scene.speakers)
        speakers2 = len(result2.scene.speakers)

        return {
            "embedding_similarity": similarity,
            "common_events": list(common_events),
            "unique_to_audio1": list(unique_to_1),
            "unique_to_audio2": list(unique_to_2),
            "audio1_speakers": speakers1,
            "audio2_speakers": speakers2,
            "audio1_duration": result1.scene.duration,
            "audio2_duration": result2.scene.duration,
            "audio1_description": result1.scene.describe(),
            "audio2_description": result2.scene.describe(),
        }

    # ==================== Utilities ====================

    def get_stats(self) -> dict[str, Any]:
        """Get auditory cortex statistics."""
        stats = {**self._stats}

        if stats["audio_processed"] > 0:
            stats["avg_processing_time_ms"] = (
                stats["total_processing_time_ms"] / stats["audio_processed"]
            )

        stats["perception"] = self._perception.get_stats()

        if self._memory:
            stats["memory"] = self._memory.get_stats()

        stats["tts_available"] = self._tts_available
        stats["capabilities"] = self._perception.get_capabilities()

        return stats

    @property
    def memory(self) -> Optional[AudioMemory]:
        """Get the audio memory system."""
        return self._memory

    @property
    def perception(self) -> AudioPerception:
        """Get the audio perception system."""
        return self._perception
