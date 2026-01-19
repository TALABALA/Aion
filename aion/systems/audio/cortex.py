"""
AION Auditory Cortex

Complete audio perception and understanding system providing:
- Speech recognition with speaker diarization (Whisper-X when available)
- Audio event detection and scene understanding
- Speaker identification and verification
- Text-to-speech generation with voice cloning (XTTS-v2 when available)
- Audio memory and retrieval
- Audio-based reasoning and QA
- Speech emotion recognition (wav2vec2-emotion when available)
- Audio source separation (Demucs when available)
- Audio language model integration (Qwen-Audio when available)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioComparisonResult,
    AudioEvent,
    AudioMemoryEntry,
    AudioScene,
    AudioSegment,
    EmotionalTone,
    MusicAnalysis,
    Speaker,
    SpeechStyle,
    SynthesisRequest,
    SynthesisResult,
    Transcript,
    VoiceCharacteristics,
)
from aion.systems.audio.perception import AudioPerception, PerceptionConfig
from aion.systems.audio.memory import AudioMemory, AudioSearchResult

# Try to import SOTA models
_WHISPERX_AVAILABLE = False
_EMOTION_AVAILABLE = False
_DEMUCS_AVAILABLE = False
_XTTS_AVAILABLE = False
_AUDIO_LLM_AVAILABLE = False

try:
    from aion.systems.audio.sota_models import (
        WhisperXTranscriber,
        WhisperXConfig,
    )
    _WHISPERX_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        SpeechEmotionRecognizer,
        EmotionResult,
    )
    _EMOTION_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        AudioSourceSeparator,
        SeparatedSources,
    )
    _DEMUCS_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        XTTSSynthesizer,
        XTTSConfig,
    )
    _XTTS_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import AudioLanguageModel
    _AUDIO_LLM_AVAILABLE = True
except ImportError:
    pass

logger = structlog.get_logger(__name__)


@dataclass
class AuditoryCortexConfig:
    """Configuration for the Auditory Cortex."""
    # Perception settings
    whisper_model: str = "openai/whisper-large-v3"
    event_detection_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    clap_model: str = "laion/larger_clap_general"

    # TTS settings
    tts_model: str = "suno/bark-small"
    enable_tts: bool = True

    # Diarization settings
    enable_diarization: bool = True
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # Memory settings
    enable_memory: bool = True
    memory_embedding_dim: int = 512
    memory_max_entries: int = 50000
    memory_index_path: Optional[str] = None

    # Processing settings
    device: str = "auto"
    target_sample_rate: int = 16000
    max_audio_duration: float = 600.0  # 10 minutes

    # Analysis settings
    enable_music_analysis: bool = True
    event_threshold: float = 0.3

    # SOTA model settings (auto-enabled when packages available)
    use_whisperx: bool = True  # Use Whisper-X for better alignment
    use_emotion_recognition: bool = True  # Use wav2vec2 emotion model
    use_source_separation: bool = True  # Use Demucs for source separation
    use_xtts: bool = True  # Use XTTS-v2 for high-quality TTS
    use_audio_llm: bool = True  # Use Qwen-Audio for direct understanding

    # Whisper-X specific settings
    whisperx_compute_type: str = "float16"  # "float16", "int8", "float32"
    whisperx_batch_size: int = 16

    # XTTS specific settings
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Source separation settings
    demucs_model: str = "htdemucs"
    separate_before_transcribe: bool = True  # Separate vocals before ASR

    # HuggingFace token for pyannote (required for diarization)
    hf_token: Optional[str] = None


@dataclass
class AnalysisResult:
    """Result of audio analysis."""
    audio_id: str
    scene: AudioScene
    transcript: Optional[Transcript] = None
    speakers: list[Speaker] = field(default_factory=list)
    memory_id: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_id": self.audio_id,
            "scene": self.scene.to_dict(),
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "speakers": [s.to_dict() for s in self.speakers],
            "memory_id": self.memory_id,
            "processing_time_ms": self.processing_time_ms,
        }


class AuditoryCortex:
    """
    AION Auditory Cortex - Complete audio perception system.

    Provides:
    - Speech recognition with diarization
    - Audio event detection and scene understanding
    - Speaker identification and verification
    - Text-to-speech generation with voice cloning
    - Audio memory and retrieval
    - Audio-based reasoning and QA

    Example:
        ```python
        cortex = AuditoryCortex()
        await cortex.initialize()

        # Transcribe speech
        transcript = await cortex.transcribe("audio.wav")

        # Full scene understanding
        scene = await cortex.understand_scene("audio.wav")

        # Generate speech
        audio = await cortex.synthesize("Hello, world!")

        # Store in memory
        memory_id = await cortex.remember("audio.wav", context="Meeting notes")

        # Recall similar
        results = await cortex.recall_similar("What was discussed?")
        ```
    """

    def __init__(
        self,
        config: Optional[AuditoryCortexConfig] = None,
        enable_memory: bool = True,
        device: str = "auto",
        whisper_model: str = "openai/whisper-large-v3",
        enable_diarization: bool = True,
        enable_tts: bool = True,
    ):
        """
        Initialize the Auditory Cortex.

        Args:
            config: Full configuration object
            enable_memory: Enable audio memory system
            device: Device for inference ("cpu", "cuda", "mps", "auto")
            whisper_model: Whisper model to use for ASR
            enable_diarization: Enable speaker diarization
            enable_tts: Enable text-to-speech
        """
        # Build config from parameters if not provided
        if config is None:
            config = AuditoryCortexConfig(
                enable_memory=enable_memory,
                device=device,
                whisper_model=whisper_model,
                enable_diarization=enable_diarization,
                enable_tts=enable_tts,
            )

        self.config = config

        # Initialize perception (fallback for non-SOTA)
        perception_config = PerceptionConfig(
            whisper_model=config.whisper_model,
            enable_diarization=config.enable_diarization,
            diarization_model=config.diarization_model,
            event_detection_model=config.event_detection_model,
            speaker_embedding_model=config.speaker_embedding_model,
            clap_model=config.clap_model,
            enable_music_analysis=config.enable_music_analysis,
            device=config.device,
            target_sample_rate=config.target_sample_rate,
            event_threshold=config.event_threshold,
        )
        self._perception = AudioPerception(perception_config)

        # Initialize memory if enabled
        self._memory: Optional[AudioMemory] = None
        if config.enable_memory:
            self._memory = AudioMemory(
                embedding_dim=config.memory_embedding_dim,
                max_entries=config.memory_max_entries,
                index_path=config.memory_index_path,
            )

        # SOTA models (initialized lazily)
        self._whisperx: Optional[Any] = None
        self._emotion_recognizer: Optional[Any] = None
        self._source_separator: Optional[Any] = None
        self._xtts: Optional[Any] = None
        self._audio_llm: Optional[Any] = None

        # Legacy TTS model (lazy loaded, used if XTTS not available)
        self._tts_model = None
        self._tts_processor = None

        # Track which SOTA features are active
        self._sota_status = {
            "whisperx": False,
            "emotion": False,
            "demucs": False,
            "xtts": False,
            "audio_llm": False,
        }

        # Statistics
        self._stats = {
            "transcriptions": 0,
            "scene_analyses": 0,
            "events_detected": 0,
            "speakers_identified": 0,
            "speeches_synthesized": 0,
            "memories_stored": 0,
            "memories_recalled": 0,
            "total_audio_processed_seconds": 0.0,
            "emotions_analyzed": 0,
            "sources_separated": 0,
            "audio_llm_queries": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize models and subsystems (lazy loading)."""
        if self._initialized:
            return

        logger.info("Initializing Auditory Cortex")

        # Initialize base perception
        await self._perception.initialize()

        # Initialize memory
        if self._memory:
            await self._memory.initialize()

        # Initialize SOTA models if available and enabled
        await self._initialize_sota_models()

        self._initialized = True
        logger.info("Auditory Cortex initialized", sota_status=self._sota_status)

    async def _initialize_sota_models(self) -> None:
        """Initialize SOTA models when available."""

        # Whisper-X: SOTA transcription with alignment
        if _WHISPERX_AVAILABLE and self.config.use_whisperx:
            try:
                whisperx_config = WhisperXConfig(
                    model_size=self.config.whisper_model.split("/")[-1] if "/" in self.config.whisper_model else "large-v3",
                    device=self.config.device,
                    compute_type=self.config.whisperx_compute_type,
                    batch_size=self.config.whisperx_batch_size,
                    enable_diarization=self.config.enable_diarization,
                    hf_token=self.config.hf_token,
                )
                self._whisperx = WhisperXTranscriber(whisperx_config)
                if await self._whisperx.initialize():
                    self._sota_status["whisperx"] = True
                    logger.info("Whisper-X initialized (SOTA transcription)")
            except Exception as e:
                logger.warning(f"Whisper-X initialization failed: {e}")

        # Emotion Recognition: wav2vec2-based
        if _EMOTION_AVAILABLE and self.config.use_emotion_recognition:
            try:
                self._emotion_recognizer = SpeechEmotionRecognizer()
                if await self._emotion_recognizer.initialize():
                    self._sota_status["emotion"] = True
                    logger.info("Speech emotion recognizer initialized")
            except Exception as e:
                logger.warning(f"Emotion recognition initialization failed: {e}")

        # Demucs: Source separation
        if _DEMUCS_AVAILABLE and self.config.use_source_separation:
            try:
                self._source_separator = AudioSourceSeparator(self.config.demucs_model)
                if await self._source_separator.initialize():
                    self._sota_status["demucs"] = True
                    logger.info("Demucs source separator initialized")
            except Exception as e:
                logger.warning(f"Demucs initialization failed: {e}")

        # XTTS: High-quality TTS with voice cloning
        if _XTTS_AVAILABLE and self.config.use_xtts and self.config.enable_tts:
            try:
                xtts_config = XTTSConfig(
                    model_name=self.config.xtts_model,
                    device=self.config.device,
                )
                self._xtts = XTTSSynthesizer(xtts_config)
                if await self._xtts.initialize():
                    self._sota_status["xtts"] = True
                    logger.info("XTTS-v2 initialized (SOTA TTS with voice cloning)")
            except Exception as e:
                logger.warning(f"XTTS initialization failed: {e}")

        # Audio LLM: Direct audio understanding
        if _AUDIO_LLM_AVAILABLE and self.config.use_audio_llm:
            try:
                self._audio_llm = AudioLanguageModel()
                if await self._audio_llm.initialize():
                    self._sota_status["audio_llm"] = True
                    logger.info("Audio LLM initialized (direct audio understanding)")
            except Exception as e:
                logger.warning(f"Audio LLM initialization failed: {e}")

    async def shutdown(self) -> None:
        """Cleanup resources including all SOTA models."""
        logger.info("Shutting down Auditory Cortex", sota_status=self._sota_status)

        # Shutdown base perception
        await self._perception.shutdown()

        # Shutdown memory
        if self._memory:
            await self._memory.shutdown()

        # Clear legacy TTS model
        self._tts_model = None
        self._tts_processor = None

        # Shutdown SOTA models
        if self._whisperx:
            try:
                if hasattr(self._whisperx, 'shutdown'):
                    await self._whisperx.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down Whisper-X: {e}")
            self._whisperx = None

        if self._emotion_recognizer:
            try:
                if hasattr(self._emotion_recognizer, 'shutdown'):
                    await self._emotion_recognizer.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down emotion recognizer: {e}")
            self._emotion_recognizer = None

        if self._source_separator:
            try:
                if hasattr(self._source_separator, 'shutdown'):
                    await self._source_separator.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down source separator: {e}")
            self._source_separator = None

        if self._xtts:
            try:
                if hasattr(self._xtts, 'shutdown'):
                    await self._xtts.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down XTTS: {e}")
            self._xtts = None

        if self._audio_llm:
            try:
                if hasattr(self._audio_llm, 'shutdown'):
                    await self._audio_llm.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down Audio LLM: {e}")
            self._audio_llm = None

        # Reset SOTA status
        self._sota_status = {
            "whisperx": False,
            "emotion": False,
            "demucs": False,
            "xtts": False,
            "audio_llm": False,
        }

        self._initialized = False
        logger.info("Auditory Cortex shutdown complete")

    # ========================
    # Core Perception
    # ========================

    async def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        language: Optional[str] = None,
        enable_diarization: bool = True,
        enable_timestamps: bool = True,
        separate_vocals: Optional[bool] = None,
    ) -> Transcript:
        """
        Transcribe speech to text with optional speaker diarization.

        Uses Whisper-X when available for:
        - 16x faster batched inference
        - Phoneme-level forced alignment
        - Accurate word timestamps

        Args:
            audio: Audio source (path, bytes, numpy array, or AudioSegment)
            language: Language code (auto-detect if None)
            enable_diarization: Perform speaker diarization
            enable_timestamps: Include word-level timestamps
            separate_vocals: Extract vocals before transcribing (better for noisy audio)
                           Defaults to config.separate_before_transcribe if Demucs available

        Returns:
            Transcript with segments, speakers, and timing
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Optionally separate vocals first for cleaner transcription
        audio_to_transcribe = audio
        if separate_vocals is None:
            separate_vocals = self.config.separate_before_transcribe and self._sota_status["demucs"]

        if separate_vocals and self._sota_status["demucs"] and self._source_separator:
            try:
                logger.debug("Separating vocals before transcription")
                separated = await self._source_separator.separate(audio)
                if separated.vocals is not None:
                    audio_to_transcribe = AudioSegment(
                        waveform=separated.vocals,
                        sample_rate=separated.sample_rate,
                        channels=1,
                    )
                    self._stats["sources_separated"] += 1
            except Exception as e:
                logger.warning(f"Source separation failed, using original audio: {e}")

        # Use Whisper-X if available (SOTA transcription)
        if self._sota_status["whisperx"] and self._whisperx:
            try:
                transcript = await self._whisperx.transcribe(
                    audio_to_transcribe,
                    language=language,
                    enable_diarization=enable_diarization and self.config.enable_diarization,
                )
                logger.debug("Used Whisper-X for transcription (SOTA)")
            except Exception as e:
                logger.warning(f"Whisper-X failed, falling back to base Whisper: {e}")
                # Fallback to base perception
                transcript = await self._perception.transcribe(
                    audio_to_transcribe,
                    language=language,
                    return_timestamps=enable_timestamps,
                )
                if enable_diarization and self.config.enable_diarization:
                    speakers = await self._perception.diarize(audio_to_transcribe)
                    if speakers:
                        transcript = await self._perception.align_transcript_with_speakers(
                            transcript, speakers
                        )
        else:
            # Base perception transcription
            transcript = await self._perception.transcribe(
                audio_to_transcribe,
                language=language,
                return_timestamps=enable_timestamps,
            )

            # Diarize if requested
            if enable_diarization and self.config.enable_diarization:
                speakers = await self._perception.diarize(audio_to_transcribe)
                if speakers:
                    transcript = await self._perception.align_transcript_with_speakers(
                        transcript, speakers
                    )

        self._stats["transcriptions"] += 1
        self._stats["total_audio_processed_seconds"] += transcript.duration

        logger.debug(
            "Transcription complete",
            duration=transcript.duration,
            speakers=len(transcript.speakers),
            processing_time_ms=(time.time() - start_time) * 1000,
            used_whisperx=self._sota_status["whisperx"],
            used_source_separation=separate_vocals and self._sota_status["demucs"],
        )

        return transcript

    async def detect_events(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        threshold: Optional[float] = None,
    ) -> list[AudioEvent]:
        """
        Detect audio events (speech, music, environmental sounds).

        Args:
            audio: Audio source
            threshold: Detection confidence threshold

        Returns:
            List of AudioEvent objects
        """
        if not self._initialized:
            await self.initialize()

        events = await self._perception.detect_events(
            audio,
            threshold=threshold or self.config.event_threshold,
        )

        self._stats["events_detected"] += len(events)

        return events

    async def understand_scene(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        store_in_memory: bool = False,
        context: str = "",
    ) -> AudioScene:
        """
        Full audio scene understanding combining all capabilities.

        Args:
            audio: Audio source
            store_in_memory: Store result in audio memory
            context: Context for memory storage

        Returns:
            Complete AudioScene analysis
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Full scene analysis
        scene = await self._perception.understand_scene(
            audio,
            enable_transcription=True,
            enable_diarization=self.config.enable_diarization,
            enable_events=True,
            enable_music=self.config.enable_music_analysis,
        )

        # Store in memory if requested
        if store_in_memory and self._memory and scene.scene_embedding is not None:
            await self._memory.store(
                embedding=scene.scene_embedding,
                audio_scene=scene,
                transcript=scene.transcript,
                context=context,
                duration=scene.duration,
            )
            self._stats["memories_stored"] += 1

        self._stats["scene_analyses"] += 1
        self._stats["total_audio_processed_seconds"] += scene.duration

        return scene

    async def analyze(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        store_in_memory: bool = False,
        context: str = "",
    ) -> AnalysisResult:
        """
        Complete audio analysis with all available information.

        Args:
            audio: Audio source
            store_in_memory: Store result in memory
            context: Context for memory

        Returns:
            AnalysisResult with scene, transcript, and speakers
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        audio_id = str(uuid.uuid4())

        scene = await self.understand_scene(audio, store_in_memory=False)
        scene.audio_id = audio_id

        memory_id = None
        if store_in_memory and self._memory and scene.scene_embedding is not None:
            entry = await self._memory.store(
                embedding=scene.scene_embedding,
                audio_scene=scene,
                transcript=scene.transcript,
                context=context,
                duration=scene.duration,
            )
            memory_id = entry.id
            self._stats["memories_stored"] += 1

        return AnalysisResult(
            audio_id=audio_id,
            scene=scene,
            transcript=scene.transcript,
            speakers=scene.speakers,
            memory_id=memory_id,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    # ========================
    # Speaker Operations
    # ========================

    async def identify_speaker(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        known_speakers: Optional[list[Speaker]] = None,
    ) -> tuple[Optional[Speaker], float]:
        """
        Identify speaker from audio.

        Args:
            audio: Audio sample (should contain clear speech)
            known_speakers: List of known speakers to match against

        Returns:
            Tuple of (matched Speaker or None, confidence score)
        """
        if not self._initialized:
            await self.initialize()

        # Compute speaker embedding
        embedding = await self._perception.compute_speaker_embedding(audio)
        if embedding is None:
            return None, 0.0

        # Try registered speakers in memory
        if self._memory:
            speaker, similarity = await self._memory.identify_speaker(embedding)
            if speaker:
                self._stats["speakers_identified"] += 1
                return speaker, similarity

        # Try provided speakers
        if known_speakers:
            best_speaker = None
            best_similarity = 0.0

            for speaker in known_speakers:
                if speaker.embedding is not None:
                    sim = float(np.dot(embedding, speaker.embedding) /
                               (np.linalg.norm(embedding) * np.linalg.norm(speaker.embedding) + 1e-10))
                    if sim > best_similarity:
                        best_similarity = sim
                        best_speaker = speaker

            if best_similarity > 0.75:
                self._stats["speakers_identified"] += 1
                return best_speaker, best_similarity

        return None, 0.0

    async def register_speaker(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        name: str,
    ) -> Speaker:
        """
        Register a new speaker from sample audio.

        Args:
            audio: Audio sample with clear speech from the speaker
            name: Name to assign to the speaker

        Returns:
            Registered Speaker object
        """
        if not self._initialized:
            await self.initialize()

        # Compute speaker embedding
        embedding = await self._perception.compute_speaker_embedding(audio)

        # Analyze voice characteristics
        voice_chars = await self._perception.analyze_voice(audio)

        # Create speaker
        speaker = Speaker(
            name=name,
            embedding=embedding,
            confidence=0.9,
        )

        # Register in memory
        if self._memory:
            await self._memory.register_speaker(speaker, name)

        logger.info("Speaker registered", speaker_id=speaker.id, name=name)
        return speaker

    async def verify_speaker(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        claimed_speaker: Speaker,
    ) -> tuple[bool, float]:
        """
        Verify if audio matches claimed speaker.

        Args:
            audio: Audio sample to verify
            claimed_speaker: Speaker to verify against

        Returns:
            Tuple of (is_verified, similarity_score)
        """
        if not self._initialized:
            await self.initialize()

        if claimed_speaker.embedding is None:
            return False, 0.0

        # Compute embedding for input audio
        embedding = await self._perception.compute_speaker_embedding(audio)
        if embedding is None:
            return False, 0.0

        # Compare embeddings
        similarity = float(np.dot(embedding, claimed_speaker.embedding) /
                          (np.linalg.norm(embedding) * np.linalg.norm(claimed_speaker.embedding) + 1e-10))

        threshold = 0.75
        return similarity >= threshold, similarity

    async def compare_speakers(
        self,
        audio1: Union[str, Path, np.ndarray, AudioSegment],
        audio2: Union[str, Path, np.ndarray, AudioSegment],
    ) -> tuple[bool, float]:
        """
        Compare two audio samples for speaker similarity.

        Args:
            audio1: First audio sample
            audio2: Second audio sample

        Returns:
            Tuple of (is_same_speaker, similarity_score)
        """
        if not self._initialized:
            await self.initialize()

        return await self._perception.compare_speakers(audio1, audio2)

    # ========================
    # Speech Generation (TTS)
    # ========================

    async def _ensure_tts_loaded(self) -> None:
        """Ensure TTS model is loaded."""
        if self._tts_model is not None:
            return

        if not self.config.enable_tts:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_tts)

    def _load_tts(self) -> None:
        """Load TTS model (blocking)."""
        try:
            from transformers import AutoProcessor, BarkModel
            import torch

            logger.info("Loading TTS model", model=self.config.tts_model)

            self._tts_processor = AutoProcessor.from_pretrained(self.config.tts_model)
            self._tts_model = BarkModel.from_pretrained(
                self.config.tts_model,
                torch_dtype=torch.float16 if self._perception.device == "cuda" else torch.float32,
            )

            if self._perception.device != "cpu":
                self._tts_model = self._tts_model.to(self._perception.device)

            logger.info("TTS model loaded successfully")

        except ImportError as e:
            logger.warning(f"TTS model not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load TTS model: {e}")

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
    ) -> AudioSegment:
        """
        Generate speech from text.

        Uses XTTS-v2 when available for:
        - High-quality neural TTS
        - 17 language support
        - Emotional control
        - Natural prosody

        Args:
            text: Text to synthesize
            voice: Voice preset or speaker ID
            language: Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, hi)
            speed: Speech speed (0.5-2.0)
            emotion: Emotional style (happy, sad, angry, fearful, neutral) - XTTS only

        Returns:
            AudioSegment with synthesized speech
        """
        if not self._initialized:
            await self.initialize()

        if not self.config.enable_tts:
            logger.warning("TTS is disabled")
            return AudioSegment(
                sample_rate=self.config.target_sample_rate,
                metadata={"error": "TTS disabled"},
            )

        start_time = time.time()

        # Use XTTS if available (SOTA TTS)
        if self._sota_status["xtts"] and self._xtts:
            try:
                result = await self._xtts.synthesize(
                    text=text,
                    language=language,
                    speaker=voice,
                    speed=speed,
                    emotion=emotion,
                )
                audio = AudioSegment(
                    waveform=result.waveform,
                    sample_rate=result.sample_rate,
                    channels=1,
                    start_time=0.0,
                    end_time=len(result.waveform) / result.sample_rate,
                    metadata={
                        "text": text,
                        "voice": voice,
                        "language": language,
                        "speed": speed,
                        "emotion": emotion,
                        "engine": "xtts_v2",
                    },
                )
                self._stats["speeches_synthesized"] += 1
                logger.debug(
                    "Speech synthesized with XTTS-v2",
                    duration=audio.duration,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
                return audio
            except Exception as e:
                logger.warning(f"XTTS synthesis failed, falling back to Bark: {e}")

        # Fallback to Bark TTS
        await self._ensure_tts_loaded()

        if self._tts_model is None:
            logger.warning("TTS model not loaded, returning empty audio")
            return AudioSegment(
                sample_rate=self.config.target_sample_rate,
                metadata={"error": "TTS model not available"},
            )

        loop = asyncio.get_event_loop()
        waveform, sr = await loop.run_in_executor(
            None,
            lambda: self._synthesize_sync(text, voice, language, speed)
        )

        audio = AudioSegment(
            waveform=waveform,
            sample_rate=sr,
            channels=1,
            start_time=0.0,
            end_time=len(waveform) / sr,
            metadata={
                "text": text,
                "voice": voice,
                "language": language,
                "speed": speed,
                "engine": "bark",
            },
        )

        self._stats["speeches_synthesized"] += 1

        logger.debug(
            "Speech synthesized with Bark",
            duration=audio.duration,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        return audio

    def _synthesize_sync(
        self,
        text: str,
        voice: Optional[str],
        language: str,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Synchronous speech synthesis."""
        import torch

        # Prepare inputs
        inputs = self._tts_processor(
            text=text,
            voice_preset=voice or "v2/en_speaker_6",
            return_tensors="pt",
        )

        if self._perception.device != "cpu":
            inputs = {k: v.to(self._perception.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            audio_array = self._tts_model.generate(**inputs)

        # Get waveform
        waveform = audio_array.cpu().numpy().squeeze()

        # Bark generates at 24kHz
        sample_rate = 24000

        # Adjust speed if needed (simple time-stretching)
        if speed != 1.0:
            try:
                import librosa
                waveform = librosa.effects.time_stretch(waveform, rate=speed)
            except ImportError:
                pass

        return waveform.astype(np.float32), sample_rate

    async def clone_voice(
        self,
        reference_audio: Union[str, Path, np.ndarray, AudioSegment],
        text: str,
        language: str = "en",
        emotion: Optional[str] = None,
    ) -> AudioSegment:
        """
        Generate speech in a cloned voice.

        Uses XTTS-v2 for high-quality voice cloning from just 3 seconds of audio.

        Args:
            reference_audio: Reference audio for voice cloning (3+ seconds recommended)
            text: Text to synthesize
            language: Target language code
            emotion: Emotional style (happy, sad, angry, fearful, neutral)

        Returns:
            AudioSegment with synthesized speech in the cloned voice
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Use XTTS for voice cloning (requires XTTS to be available)
        if self._sota_status["xtts"] and self._xtts:
            try:
                # Load reference audio if needed
                if isinstance(reference_audio, (str, Path)):
                    ref_waveform, ref_sr = await self._perception.load_audio(reference_audio)
                elif isinstance(reference_audio, AudioSegment):
                    ref_waveform = reference_audio.waveform
                    ref_sr = reference_audio.sample_rate
                else:
                    ref_waveform = reference_audio
                    ref_sr = self.config.target_sample_rate

                result = await self._xtts.clone_voice(
                    text=text,
                    reference_audio=ref_waveform,
                    reference_sr=ref_sr,
                    language=language,
                    emotion=emotion,
                )

                audio = AudioSegment(
                    waveform=result.waveform,
                    sample_rate=result.sample_rate,
                    channels=1,
                    start_time=0.0,
                    end_time=len(result.waveform) / result.sample_rate,
                    metadata={
                        "text": text,
                        "language": language,
                        "emotion": emotion,
                        "engine": "xtts_v2_clone",
                        "voice_cloning": True,
                    },
                )

                self._stats["speeches_synthesized"] += 1
                logger.debug(
                    "Voice cloned with XTTS-v2",
                    duration=audio.duration,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
                return audio

            except Exception as e:
                logger.warning(f"XTTS voice cloning failed: {e}")

        # Fallback to standard synthesis without cloning
        logger.warning("Voice cloning requires XTTS-v2, using standard synthesis")
        return await self.synthesize(text, language=language)

    # ========================
    # Memory Operations
    # ========================

    async def remember(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        context: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
    ) -> str:
        """
        Store audio in memory.

        Args:
            audio: Audio to store
            context: Context/description for the audio
            importance: Importance score (0-1)
            tags: Organizational tags

        Returns:
            Memory entry ID
        """
        if not self._initialized:
            await self.initialize()

        if not self._memory:
            raise RuntimeError("Memory is not enabled")

        # Analyze audio
        scene = await self.understand_scene(audio)

        # Get or compute embedding
        if scene.scene_embedding is None:
            embedding = await self._perception.compute_audio_embedding(audio)
        else:
            embedding = scene.scene_embedding

        if embedding is None:
            embedding = np.zeros(self.config.memory_embedding_dim, dtype=np.float32)

        # Compute audio hash
        if isinstance(audio, AudioSegment) and audio.waveform is not None:
            audio_hash = self._perception.compute_audio_hash(audio.waveform)
        elif isinstance(audio, np.ndarray):
            audio_hash = self._perception.compute_audio_hash(audio)
        else:
            waveform, _ = await self._perception.load_audio(audio)
            audio_hash = self._perception.compute_audio_hash(waveform)

        # Store
        entry = await self._memory.store(
            embedding=embedding,
            audio_scene=scene,
            transcript=scene.transcript,
            context=context or "",
            duration=scene.duration,
            importance=importance,
            tags=tags,
            audio_hash=audio_hash,
        )

        self._stats["memories_stored"] += 1

        logger.debug("Audio stored in memory", memory_id=entry.id)
        return entry.id

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
            List of AudioMemoryEntry objects
        """
        if not self._initialized:
            await self.initialize()

        if not self._memory:
            raise RuntimeError("Memory is not enabled")

        # Convert query to embedding
        if isinstance(query, str):
            # Text query - use CLAP text encoder
            embedding = await self._perception.compute_text_embedding(query)
            if embedding is None:
                # Fallback to transcript search
                results = await self._memory.search_by_transcript(query, limit)
                self._stats["memories_recalled"] += len(results)
                return [r.entry for r in results]
        else:
            embedding = query

        # Search
        results = await self._memory.search_by_similarity(embedding, limit)
        self._stats["memories_recalled"] += len(results)

        return [r.entry for r in results]

    async def recall_by_speaker(
        self,
        speaker: Speaker,
        limit: int = 10,
    ) -> list[AudioMemoryEntry]:
        """
        Retrieve audio by speaker.

        Args:
            speaker: Speaker to search for
            limit: Maximum results

        Returns:
            List of AudioMemoryEntry objects
        """
        if not self._initialized:
            await self.initialize()

        if not self._memory:
            raise RuntimeError("Memory is not enabled")

        if speaker.embedding is None:
            return []

        results = await self._memory.search_by_speaker(speaker.embedding, limit)
        self._stats["memories_recalled"] += len(results)

        return [r.entry for r in results]

    async def recall_by_tags(
        self,
        tags: list[str],
        require_all: bool = False,
        limit: int = 10,
    ) -> list[AudioMemoryEntry]:
        """
        Retrieve audio by tags.

        Args:
            tags: Tags to search for
            require_all: Require all tags to match
            limit: Maximum results

        Returns:
            List of AudioMemoryEntry objects
        """
        if not self._initialized:
            await self.initialize()

        if not self._memory:
            raise RuntimeError("Memory is not enabled")

        results = await self._memory.search_by_tags(tags, require_all, limit)
        self._stats["memories_recalled"] += len(results)

        return [r.entry for r in results]

    # ========================
    # Audio Reasoning (Integration point for LLM)
    # ========================

    async def answer_question(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        question: str,
    ) -> str:
        """
        Answer a question about audio content.

        Note: This is a basic implementation. Full reasoning
        requires LLM integration (see reasoning.py).

        Args:
            audio: Audio source
            question: Question to answer

        Returns:
            Answer string
        """
        if not self._initialized:
            await self.initialize()

        # Get scene understanding
        scene = await self.understand_scene(audio)

        # Basic question answering based on scene
        question_lower = question.lower()

        if "who" in question_lower or "speaker" in question_lower:
            if scene.speakers:
                speaker_info = ", ".join(
                    f"{s.name or f'Speaker {i+1}'}"
                    for i, s in enumerate(scene.speakers)
                )
                return f"The speakers in the audio are: {speaker_info}"
            return "No speakers were identified in the audio."

        if "what" in question_lower and "said" in question_lower:
            if scene.transcript:
                return f"The following was said: {scene.transcript.text}"
            return "No speech was detected in the audio."

        if "music" in question_lower:
            if scene.has_music and scene.music_analysis:
                ma = scene.music_analysis
                return (f"Music was detected at {ma.tempo:.0f} BPM in {ma.key or 'unknown key'}. "
                       f"The mood is {ma.mood or 'moderate'}.")
            return "No music was detected in the audio."

        if "event" in question_lower or "sound" in question_lower:
            if scene.events:
                event_list = ", ".join(e.label for e in scene.events[:5])
                return f"The following sounds were detected: {event_list}"
            return "No specific sounds were detected."

        if "emotion" in question_lower or "tone" in question_lower:
            if scene.emotional_tone:
                return f"The emotional tone is {scene.emotional_tone.value}."
            return "Unable to determine the emotional tone."

        # Default: return scene description
        return scene.describe()

    async def summarize(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> str:
        """
        Generate a summary of audio content.

        Args:
            audio: Audio source

        Returns:
            Summary string
        """
        if not self._initialized:
            await self.initialize()

        scene = await self.understand_scene(audio)
        return scene.describe()

    async def compare(
        self,
        audio1: Union[str, Path, np.ndarray, AudioSegment],
        audio2: Union[str, Path, np.ndarray, AudioSegment],
    ) -> AudioComparisonResult:
        """
        Compare two audio samples.

        Args:
            audio1: First audio
            audio2: Second audio

        Returns:
            AudioComparisonResult
        """
        if not self._initialized:
            await self.initialize()

        # Analyze both
        scene1 = await self.understand_scene(audio1)
        scene2 = await self.understand_scene(audio2)

        # Compute similarities
        acoustic_similarity = 0.0
        if scene1.scene_embedding is not None and scene2.scene_embedding is not None:
            acoustic_similarity = float(np.dot(scene1.scene_embedding, scene2.scene_embedding) /
                                        (np.linalg.norm(scene1.scene_embedding) *
                                         np.linalg.norm(scene2.scene_embedding) + 1e-10))

        content_similarity = 0.0
        if scene1.transcript and scene2.transcript:
            words1 = set(scene1.transcript.text.lower().split())
            words2 = set(scene2.transcript.text.lower().split())
            if words1 and words2:
                content_similarity = len(words1 & words2) / len(words1 | words2)

        # Check speaker similarity
        speaker_similarity = 0.0
        is_same_speaker, speaker_sim = await self._perception.compare_speakers(audio1, audio2)
        speaker_similarity = speaker_sim

        # Find common elements
        common_events = []
        events1 = {e.label for e in scene1.events}
        events2 = {e.label for e in scene2.events}
        common_events = list(events1 & events2)

        # Find differences
        differences = []
        if scene1.scene_type != scene2.scene_type:
            differences.append(f"Scene type: {scene1.scene_type} vs {scene2.scene_type}")
        if scene1.has_music != scene2.has_music:
            differences.append(f"Music: {'present' if scene1.has_music else 'absent'} vs {'present' if scene2.has_music else 'absent'}")
        if abs(scene1.duration - scene2.duration) > 1.0:
            differences.append(f"Duration: {scene1.duration:.1f}s vs {scene2.duration:.1f}s")

        overall_similarity = (acoustic_similarity * 0.5 + content_similarity * 0.3 + speaker_similarity * 0.2)

        return AudioComparisonResult(
            audio1_id=scene1.id,
            audio2_id=scene2.id,
            overall_similarity=overall_similarity,
            acoustic_similarity=acoustic_similarity,
            content_similarity=content_similarity,
            speaker_similarity=speaker_similarity,
            differences=differences,
            common_events=common_events,
        )

    # ========================
    # Music Analysis
    # ========================

    async def analyze_music(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> MusicAnalysis:
        """
        Analyze music content.

        Args:
            audio: Audio source

        Returns:
            MusicAnalysis with tempo, key, mood, etc.
        """
        if not self._initialized:
            await self.initialize()

        return await self._perception.analyze_music(audio)

    # ========================
    # SOTA: Emotion Recognition
    # ========================

    async def analyze_emotion(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> dict[str, Any]:
        """
        Analyze emotional content in speech.

        Uses wav2vec2-large-robust emotion model for dimensional emotion
        analysis (arousal, valence, dominance) mapped to categorical emotions.

        Args:
            audio: Audio source containing speech

        Returns:
            Dictionary with emotion analysis:
            - primary_emotion: str (angry, happy, sad, neutral, fearful, disgusted, surprised)
            - confidence: float
            - arousal: float (-1 to 1, low to high energy)
            - valence: float (-1 to 1, negative to positive)
            - dominance: float (-1 to 1, submissive to dominant)
            - all_emotions: dict[str, float] (probability distribution)
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        if self._sota_status["emotion"] and self._emotion_recognizer:
            try:
                result = await self._emotion_recognizer.analyze(audio)
                self._stats["emotions_analyzed"] += 1

                logger.debug(
                    "Emotion analysis complete",
                    primary_emotion=result.primary_emotion,
                    confidence=result.confidence,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

                return {
                    "primary_emotion": result.primary_emotion,
                    "confidence": result.confidence,
                    "arousal": result.arousal,
                    "valence": result.valence,
                    "dominance": result.dominance,
                    "all_emotions": result.all_emotions,
                }
            except Exception as e:
                logger.warning(f"Emotion analysis failed: {e}")

        # Fallback to basic heuristic-based emotion detection
        scene = await self.understand_scene(audio)
        return {
            "primary_emotion": scene.emotional_tone.value if scene.emotional_tone else "neutral",
            "confidence": 0.5,
            "arousal": 0.0,
            "valence": 0.0,
            "dominance": 0.0,
            "all_emotions": {},
            "note": "Basic heuristic analysis - install wav2vec2-emotion for SOTA results",
        }

    # ========================
    # SOTA: Source Separation
    # ========================

    async def separate_sources(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        model: Optional[str] = None,
    ) -> dict[str, AudioSegment]:
        """
        Separate audio into component sources using Demucs.

        Separates audio into:
        - vocals: Human voice
        - drums: Percussion instruments
        - bass: Bass instruments
        - other: Other instruments

        Args:
            audio: Audio source to separate
            model: Demucs model to use (default: htdemucs)

        Returns:
            Dictionary of separated AudioSegments:
            - "vocals": Isolated voice track
            - "drums": Isolated drums track
            - "bass": Isolated bass track
            - "other": Other instruments track
        """
        if not self._initialized:
            await self.initialize()

        if not self._sota_status["demucs"] or not self._source_separator:
            raise RuntimeError(
                "Source separation requires Demucs. Install with: pip install demucs"
            )

        start_time = time.time()

        try:
            separated = await self._source_separator.separate(audio, model=model)
            self._stats["sources_separated"] += 1

            result = {}
            for source_name in ["vocals", "drums", "bass", "other"]:
                waveform = getattr(separated, source_name, None)
                if waveform is not None:
                    result[source_name] = AudioSegment(
                        waveform=waveform,
                        sample_rate=separated.sample_rate,
                        channels=1,
                        start_time=0.0,
                        end_time=len(waveform) / separated.sample_rate,
                        metadata={"source": source_name, "model": separated.model_used},
                    )

            logger.debug(
                "Source separation complete",
                sources=list(result.keys()),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            return result

        except Exception as e:
            logger.error(f"Source separation failed: {e}")
            raise

    async def extract_vocals(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> AudioSegment:
        """
        Extract vocals from audio (shortcut for separate_sources).

        Useful for:
        - Improving transcription quality in noisy audio
        - Karaoke creation
        - Voice isolation

        Args:
            audio: Audio source

        Returns:
            AudioSegment containing only the vocal track
        """
        sources = await self.separate_sources(audio)
        return sources.get("vocals", AudioSegment(sample_rate=self.config.target_sample_rate))

    # ========================
    # SOTA: Audio Language Model
    # ========================

    async def query_audio(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        query: str,
    ) -> str:
        """
        Query audio content directly using Audio LLM (Qwen-Audio).

        Enables direct audio understanding without ASR bottleneck:
        - "What sounds are in this audio?"
        - "Describe the music"
        - "What is the speaker talking about?"
        - "Is this speech or music?"
        - "What emotion is expressed?"

        Args:
            audio: Audio source
            query: Natural language question about the audio

        Returns:
            Natural language answer from the Audio LLM
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        if self._sota_status["audio_llm"] and self._audio_llm:
            try:
                answer = await self._audio_llm.query(audio, query)
                self._stats["audio_llm_queries"] += 1

                logger.debug(
                    "Audio LLM query complete",
                    query=query[:50],
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

                return answer

            except Exception as e:
                logger.warning(f"Audio LLM query failed: {e}")

        # Fallback to rule-based answer_question
        logger.info("Audio LLM not available, using rule-based QA")
        return await self.answer_question(audio, query)

    async def describe_audio(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> str:
        """
        Generate a natural language description of audio content.

        Uses Audio LLM when available for rich, contextual descriptions.

        Args:
            audio: Audio source

        Returns:
            Natural language description of the audio
        """
        if self._sota_status["audio_llm"] and self._audio_llm:
            return await self.query_audio(
                audio,
                "Provide a detailed description of this audio, including any speech, "
                "music, environmental sounds, and emotional qualities you detect."
            )

        # Fallback to scene description
        scene = await self.understand_scene(audio)
        return scene.describe()

    # ========================
    # SOTA Status & Capabilities
    # ========================

    def get_sota_status(self) -> dict[str, bool]:
        """
        Get the availability status of SOTA features.

        Returns:
            Dictionary mapping feature names to availability:
            - whisperx: SOTA transcription with forced alignment
            - emotion: wav2vec2-based emotion recognition
            - demucs: Source separation
            - xtts: High-quality TTS with voice cloning
            - audio_llm: Direct audio understanding
        """
        return dict(self._sota_status)

    def get_capabilities(self) -> dict[str, Any]:
        """
        Get a summary of all available capabilities.

        Returns:
            Dictionary of capability categories and their status
        """
        return {
            "transcription": {
                "available": True,
                "sota": self._sota_status["whisperx"],
                "features": {
                    "diarization": self.config.enable_diarization,
                    "word_timestamps": True,
                    "forced_alignment": self._sota_status["whisperx"],
                    "batched_inference": self._sota_status["whisperx"],
                },
            },
            "emotion_recognition": {
                "available": self._sota_status["emotion"],
                "sota": self._sota_status["emotion"],
                "features": {
                    "dimensional": self._sota_status["emotion"],  # arousal, valence, dominance
                    "categorical": True,
                },
            },
            "source_separation": {
                "available": self._sota_status["demucs"],
                "sota": self._sota_status["demucs"],
                "features": {
                    "vocals": self._sota_status["demucs"],
                    "drums": self._sota_status["demucs"],
                    "bass": self._sota_status["demucs"],
                    "other": self._sota_status["demucs"],
                },
            },
            "text_to_speech": {
                "available": self.config.enable_tts,
                "sota": self._sota_status["xtts"],
                "features": {
                    "voice_cloning": self._sota_status["xtts"],
                    "multilingual": self._sota_status["xtts"],
                    "emotional_control": self._sota_status["xtts"],
                    "languages": 17 if self._sota_status["xtts"] else 1,
                },
            },
            "audio_understanding": {
                "available": True,
                "sota": self._sota_status["audio_llm"],
                "features": {
                    "direct_audio_query": self._sota_status["audio_llm"],
                    "scene_understanding": True,
                    "event_detection": True,
                    "music_analysis": self.config.enable_music_analysis,
                },
            },
            "memory": {
                "available": self._memory is not None,
                "features": {
                    "similarity_search": True,
                    "speaker_search": True,
                    "transcript_search": True,
                },
            },
        }

    # ========================
    # Utilities
    # ========================

    async def load_audio(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        target_sr: Optional[int] = None,
    ) -> AudioSegment:
        """
        Load audio from various sources.

        Args:
            source: Audio source
            target_sr: Target sample rate

        Returns:
            AudioSegment
        """
        waveform, sr = await self._perception.load_audio(
            source,
            target_sr=target_sr or self.config.target_sample_rate,
        )

        return AudioSegment(
            waveform=waveform,
            sample_rate=sr,
            channels=1,
            start_time=0.0,
            end_time=len(waveform) / sr,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics including SOTA model usage."""
        stats = {
            **self._stats,
            "perception": self._perception.get_stats(),
            "sota_status": dict(self._sota_status),
            "sota_features_active": sum(1 for v in self._sota_status.values() if v),
            "sota_features_total": len(self._sota_status),
        }

        if self._memory:
            stats["memory"] = self._memory.get_stats()

        # Add capability summary
        stats["capabilities"] = {
            "transcription_engine": "whisperx" if self._sota_status["whisperx"] else "whisper",
            "tts_engine": "xtts_v2" if self._sota_status["xtts"] else "bark",
            "emotion_model": "wav2vec2" if self._sota_status["emotion"] else "heuristic",
            "source_separation": "demucs" if self._sota_status["demucs"] else "none",
            "audio_understanding": "qwen_audio" if self._sota_status["audio_llm"] else "rule_based",
        }

        return stats

    @property
    def memory(self) -> Optional[AudioMemory]:
        """Access the audio memory system."""
        return self._memory

    @property
    def perception(self) -> AudioPerception:
        """Access the perception system."""
        return self._perception
