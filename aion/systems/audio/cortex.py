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

# Additional TRUE SOTA models
_EMOTION2VEC_AVAILABLE = False
_BEATS_AVAILABLE = False
_DEEPFILTER_AVAILABLE = False
_STREAMING_ASR_AVAILABLE = False
_MUSICGEN_AVAILABLE = False
_CAPTIONER_AVAILABLE = False

try:
    from aion.systems.audio.sota_models import (
        Emotion2VecRecognizer,
        Emotion2VecResult,
    )
    _EMOTION2VEC_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        BEATsEventDetector,
        AudioEventResult,
    )
    _BEATS_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        DeepFilterNetEnhancer,
        EnhancedAudio,
    )
    _DEEPFILTER_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        StreamingASR,
        StreamingTranscript,
    )
    _STREAMING_ASR_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        MusicGenerator,
        GeneratedMusic,
    )
    _MUSICGEN_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        AudioCaptioner,
        AudioCaption,
    )
    _CAPTIONER_AVAILABLE = True
except ImportError:
    pass

# Niche SOTA models
_QUALITY_AVAILABLE = False
_SELD_AVAILABLE = False
_INPAINTING_AVAILABLE = False

try:
    from aion.systems.audio.sota_models import (
        AudioQualityAssessor,
        AudioQualityResult,
    )
    _QUALITY_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        SELDDetector,
        SELDResult,
        LocalizedEvent,
    )
    _SELD_AVAILABLE = True
except ImportError:
    pass

try:
    from aion.systems.audio.sota_models import (
        AudioInpainter,
        InpaintedAudio,
    )
    _INPAINTING_AVAILABLE = True
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

    # TRUE SOTA additions
    use_emotion2vec: bool = True  # emotion2vec (better than wav2vec2)
    use_beats: bool = True  # BEATs for event detection (better than AST)
    use_deepfilter: bool = True  # DeepFilterNet speech enhancement
    use_streaming_asr: bool = False  # Streaming ASR (disabled by default)
    use_musicgen: bool = False  # MusicGen (disabled by default, resource-heavy)
    use_captioner: bool = True  # Audio captioning

    # Emotion2vec settings
    emotion2vec_model: str = "iic/emotion2vec_base_finetuned"

    # Streaming ASR settings
    streaming_model_size: str = "base"  # tiny, base, small, medium, large-v3
    streaming_vad_threshold: float = 0.5

    # MusicGen settings
    musicgen_model: str = "small"  # small, medium, large, melody

    # Speech enhancement settings
    enhance_before_transcribe: bool = False  # Apply enhancement before ASR

    # Niche SOTA additions
    use_quality_assessment: bool = True  # NISQA/DNSMOS quality metrics
    use_seld: bool = False  # Sound Event Localization and Detection (requires multi-channel)
    use_inpainting: bool = False  # AudioLDM2 inpainting (resource-heavy)

    # Quality assessment settings
    quality_model: str = "nisqa"  # "nisqa", "dnsmos", or "both"

    # SELD settings
    seld_model: str = "seld-dcase2022"
    seld_frame_duration: float = 0.5  # Analysis frame duration

    # Inpainting settings (AudioLDM2)
    inpainting_model: str = "audioldm2-large"  # "audioldm2", "audioldm2-large", "audioldm2-music"
    inpainting_steps: int = 100  # Diffusion steps


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

        # TRUE SOTA models (initialized lazily)
        self._emotion2vec: Optional[Any] = None
        self._beats: Optional[Any] = None
        self._deepfilter: Optional[Any] = None
        self._streaming_asr: Optional[Any] = None
        self._musicgen: Optional[Any] = None
        self._captioner: Optional[Any] = None

        # Niche SOTA models
        self._quality_assessor: Optional[Any] = None
        self._seld: Optional[Any] = None
        self._inpainter: Optional[Any] = None

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
            # TRUE SOTA additions
            "emotion2vec": False,
            "beats": False,
            "deepfilter": False,
            "streaming_asr": False,
            "musicgen": False,
            "captioner": False,
            # Niche SOTA additions
            "quality_assessment": False,
            "seld": False,
            "inpainting": False,
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
            # TRUE SOTA stats
            "streaming_chunks": 0,
            "enhancements": 0,
            "music_generations": 0,
            "captions_generated": 0,
            # Niche SOTA stats
            "quality_assessments": 0,
            "seld_detections": 0,
            "inpaintings": 0,
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

        # ============== TRUE SOTA MODELS ==============

        # emotion2vec: Better than wav2vec2 for emotion
        if _EMOTION2VEC_AVAILABLE and self.config.use_emotion2vec:
            try:
                self._emotion2vec = Emotion2VecRecognizer(self.config.emotion2vec_model)
                if await self._emotion2vec.initialize():
                    self._sota_status["emotion2vec"] = True
                    logger.info("emotion2vec initialized (TRUE SOTA emotion)")
            except Exception as e:
                logger.warning(f"emotion2vec initialization failed: {e}")

        # BEATs: Better than AST for audio events
        if _BEATS_AVAILABLE and self.config.use_beats:
            try:
                self._beats = BEATsEventDetector()
                if await self._beats.initialize():
                    self._sota_status["beats"] = True
                    logger.info("BEATs initialized (TRUE SOTA event detection)")
            except Exception as e:
                logger.warning(f"BEATs initialization failed: {e}")

        # DeepFilterNet: Speech enhancement
        if _DEEPFILTER_AVAILABLE and self.config.use_deepfilter:
            try:
                self._deepfilter = DeepFilterNetEnhancer()
                if await self._deepfilter.initialize():
                    self._sota_status["deepfilter"] = True
                    logger.info("DeepFilterNet initialized (speech enhancement)")
            except Exception as e:
                logger.warning(f"DeepFilterNet initialization failed: {e}")

        # Streaming ASR: Real-time transcription
        if _STREAMING_ASR_AVAILABLE and self.config.use_streaming_asr:
            try:
                self._streaming_asr = StreamingASR(
                    model_size=self.config.streaming_model_size,
                    device=self.config.device,
                    vad_threshold=self.config.streaming_vad_threshold,
                )
                if await self._streaming_asr.initialize():
                    self._sota_status["streaming_asr"] = True
                    logger.info("Streaming ASR initialized (real-time transcription)")
            except Exception as e:
                logger.warning(f"Streaming ASR initialization failed: {e}")

        # MusicGen: Music generation
        if _MUSICGEN_AVAILABLE and self.config.use_musicgen:
            try:
                self._musicgen = MusicGenerator(self.config.musicgen_model)
                if await self._musicgen.initialize():
                    self._sota_status["musicgen"] = True
                    logger.info("MusicGen initialized (music generation)")
            except Exception as e:
                logger.warning(f"MusicGen initialization failed: {e}")

        # Audio Captioning
        if _CAPTIONER_AVAILABLE and self.config.use_captioner:
            try:
                self._captioner = AudioCaptioner()
                if await self._captioner.initialize():
                    self._sota_status["captioner"] = True
                    logger.info("Audio Captioner initialized")
            except Exception as e:
                logger.warning(f"Audio Captioner initialization failed: {e}")

        # ============== NICHE SOTA MODELS ==============

        # Audio Quality Assessment (NISQA/DNSMOS)
        if _QUALITY_AVAILABLE and self.config.use_quality_assessment:
            try:
                self._quality_assessor = AudioQualityAssessor(self.config.quality_model)
                if await self._quality_assessor.initialize():
                    self._sota_status["quality_assessment"] = True
                    logger.info("Audio Quality Assessor initialized (NISQA/DNSMOS)")
            except Exception as e:
                logger.warning(f"Quality assessor initialization failed: {e}")

        # Sound Event Localization and Detection (SELD)
        if _SELD_AVAILABLE and self.config.use_seld:
            try:
                self._seld = SELDDetector(self.config.seld_model)
                if await self._seld.initialize():
                    self._sota_status["seld"] = True
                    logger.info("SELD detector initialized (spatial audio)")
            except Exception as e:
                logger.warning(f"SELD initialization failed: {e}")

        # Audio Inpainting (AudioLDM2)
        if _INPAINTING_AVAILABLE and self.config.use_inpainting:
            try:
                self._inpainter = AudioInpainter(self.config.inpainting_model)
                if await self._inpainter.initialize():
                    self._sota_status["inpainting"] = True
                    logger.info("Audio Inpainter initialized (AudioLDM2)")
            except Exception as e:
                logger.warning(f"AudioLDM2 initialization failed: {e}")

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

        # Shutdown TRUE SOTA models and niche SOTA models
        for model_name, model_attr in [
            ("emotion2vec", "_emotion2vec"),
            ("BEATs", "_beats"),
            ("DeepFilterNet", "_deepfilter"),
            ("Streaming ASR", "_streaming_asr"),
            ("MusicGen", "_musicgen"),
            ("Captioner", "_captioner"),
            # Niche SOTA
            ("Quality Assessor", "_quality_assessor"),
            ("SELD", "_seld"),
            ("Inpainter", "_inpainter"),
        ]:
            model = getattr(self, model_attr, None)
            if model:
                try:
                    if hasattr(model, 'shutdown'):
                        await model.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down {model_name}: {e}")
                setattr(self, model_attr, None)

        # Reset SOTA status
        self._sota_status = {
            "whisperx": False,
            "emotion": False,
            "demucs": False,
            "xtts": False,
            "audio_llm": False,
            "emotion2vec": False,
            "beats": False,
            "deepfilter": False,
            "streaming_asr": False,
            "musicgen": False,
            "captioner": False,
            "quality_assessment": False,
            "seld": False,
            "inpainting": False,
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
    # TRUE SOTA: Enhanced Emotion (emotion2vec)
    # ========================

    async def analyze_emotion_sota(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        return_embedding: bool = False,
    ) -> dict[str, Any]:
        """
        TRUE SOTA emotion analysis using emotion2vec.

        emotion2vec significantly outperforms wav2vec2-based models:
        - Self-supervised pretraining on 262k hours
        - State-of-the-art on all major benchmarks
        - Universal emotion representation

        Args:
            audio: Audio source
            return_embedding: Return emotion embedding

        Returns:
            Dictionary with primary_emotion, confidence, all_emotions, and optionally embedding
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Use emotion2vec if available (TRUE SOTA)
        if self._sota_status["emotion2vec"] and self._emotion2vec:
            try:
                result = await self._emotion2vec.recognize(audio, return_embedding=return_embedding)
                self._stats["emotions_analyzed"] += 1
                logger.debug(
                    "Emotion analyzed with emotion2vec (TRUE SOTA)",
                    emotion=result.primary_emotion,
                    confidence=result.confidence,
                )
                return {
                    "primary_emotion": result.primary_emotion,
                    "confidence": result.confidence,
                    "all_emotions": result.all_emotions,
                    "embedding": result.embedding if return_embedding else None,
                    "model": "emotion2vec",
                }
            except Exception as e:
                logger.warning(f"emotion2vec failed, falling back: {e}")

        # Fallback to wav2vec2-based
        return await self.analyze_emotion(audio)

    # ========================
    # TRUE SOTA: BEATs Event Detection
    # ========================

    async def detect_events_sota(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        top_k: int = 10,
        threshold: float = 0.1,
        return_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        TRUE SOTA audio event detection using BEATs.

        BEATs outperforms AST on AudioSet with 50.6% mAP.

        Args:
            audio: Audio source
            top_k: Number of top events to return
            threshold: Confidence threshold
            return_embeddings: Return audio embeddings

        Returns:
            Dictionary with events, top_events, and optionally embeddings
        """
        if not self._initialized:
            await self.initialize()

        if self._sota_status["beats"] and self._beats:
            try:
                result = await self._beats.detect_events(
                    audio, top_k=top_k, threshold=threshold, return_embeddings=return_embeddings
                )
                self._stats["events_detected"] += len(result.events)
                return {
                    "events": result.events,
                    "top_events": result.top_events,
                    "embeddings": result.embeddings,
                    "model": "beats",
                }
            except Exception as e:
                logger.warning(f"BEATs detection failed, falling back to AST: {e}")

        # Fallback to AST-based detection
        events = await self.detect_events(audio, threshold=threshold)
        return {
            "events": [{"label": e.label, "confidence": e.confidence} for e in events[:top_k]],
            "top_events": [e.label for e in events[:top_k]],
            "model": "ast",
        }

    # ========================
    # TRUE SOTA: Speech Enhancement
    # ========================

    async def enhance_speech(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> AudioSegment:
        """
        Enhance speech by removing noise using DeepFilterNet.

        Excellent for:
        - Preprocessing noisy audio before transcription
        - Improving audio quality for voice cloning
        - Cleaning up recordings

        Args:
            audio: Noisy audio input

        Returns:
            AudioSegment with cleaned speech
        """
        if not self._initialized:
            await self.initialize()

        if not self._sota_status["deepfilter"] or not self._deepfilter:
            logger.warning("DeepFilterNet not available, returning original")
            if isinstance(audio, AudioSegment):
                return audio
            waveform, sr = await self._perception.load_audio(audio)
            return AudioSegment(waveform=waveform, sample_rate=sr)

        start_time = time.time()

        try:
            result = await self._deepfilter.enhance(audio)
            self._stats["enhancements"] += 1

            logger.debug(
                "Speech enhanced with DeepFilterNet",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            return AudioSegment(
                waveform=result.waveform,
                sample_rate=result.sample_rate,
                metadata={"enhanced": True, "snr_improvement_db": result.snr_improvement_db},
            )

        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            if isinstance(audio, AudioSegment):
                return audio
            waveform, sr = await self._perception.load_audio(audio)
            return AudioSegment(waveform=waveform, sample_rate=sr)

    # ========================
    # TRUE SOTA: Streaming ASR
    # ========================

    async def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
        previous_context: str = "",
    ) -> dict[str, Any]:
        """
        Transcribe a single audio chunk for real-time streaming.

        Args:
            audio_chunk: Audio data (1-5 seconds recommended)
            sr: Sample rate
            language: Language code
            previous_context: Previous transcript for continuity

        Returns:
            Dictionary with text, is_final, confidence, timing, words
        """
        if not self._initialized:
            await self.initialize()

        if not self._sota_status["streaming_asr"] or not self._streaming_asr:
            raise RuntimeError(
                "Streaming ASR not available. Enable with use_streaming_asr=True "
                "and install faster-whisper."
            )

        result = await self._streaming_asr.transcribe_chunk(
            audio_chunk, sr, language, previous_context
        )
        self._stats["streaming_chunks"] += 1

        return {
            "text": result.text,
            "is_final": result.is_final,
            "confidence": result.confidence,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "words": result.words,
        }

    # ========================
    # TRUE SOTA: Music Generation
    # ========================

    async def generate_music(
        self,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
    ) -> AudioSegment:
        """
        Generate music from text description using MusicGen.

        Args:
            prompt: Text description (e.g., "upbeat electronic dance music with heavy bass")
            duration: Duration in seconds (max 30s for small model)
            temperature: Sampling temperature (higher = more random)

        Returns:
            AudioSegment with generated music
        """
        if not self._initialized:
            await self.initialize()

        if not self._sota_status["musicgen"] or not self._musicgen:
            raise RuntimeError(
                "MusicGen not available. Enable with use_musicgen=True "
                "and install audiocraft."
            )

        start_time = time.time()

        result = await self._musicgen.generate(
            prompt=prompt,
            duration=duration,
            temperature=temperature,
        )
        self._stats["music_generations"] += 1

        logger.debug(
            "Music generated with MusicGen",
            duration=result.duration,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        return AudioSegment(
            waveform=result.waveform,
            sample_rate=result.sample_rate,
            metadata={"prompt": prompt, "generated": True},
        )

    async def generate_music_with_melody(
        self,
        prompt: str,
        melody: Union[str, Path, np.ndarray, AudioSegment],
        duration: float = 10.0,
    ) -> AudioSegment:
        """
        Generate music conditioned on a melody.

        Requires MusicGen melody model.

        Args:
            prompt: Text description
            melody: Reference melody
            duration: Output duration

        Returns:
            AudioSegment following the melody
        """
        if not self._sota_status["musicgen"] or not self._musicgen:
            raise RuntimeError("MusicGen not available")

        # Extract melody waveform
        if isinstance(melody, AudioSegment):
            melody_wav = melody.waveform
            melody_sr = melody.sample_rate
        elif isinstance(melody, (str, Path)):
            melody_wav, melody_sr = await self._perception.load_audio(melody)
        else:
            melody_wav = melody
            melody_sr = 32000

        result = await self._musicgen.generate_with_melody(
            prompt=prompt,
            melody=melody_wav,
            melody_sr=melody_sr,
            duration=duration,
        )

        return AudioSegment(
            waveform=result.waveform,
            sample_rate=result.sample_rate,
            metadata={"prompt": prompt, "melody_conditioned": True},
        )

    # ========================
    # TRUE SOTA: Audio Captioning
    # ========================

    async def caption_audio(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        num_captions: int = 3,
    ) -> dict[str, Any]:
        """
        Generate natural language caption for audio.

        Uses CLAP-based retrieval for audio-to-text description.

        Args:
            audio: Audio input
            num_captions: Number of alternative captions

        Returns:
            Dictionary with caption, confidence, alternative_captions
        """
        if not self._initialized:
            await self.initialize()

        if self._sota_status["captioner"] and self._captioner:
            try:
                result = await self._captioner.caption(audio, num_captions=num_captions)
                self._stats["captions_generated"] += 1
                return {
                    "caption": result.caption,
                    "confidence": result.confidence,
                    "alternative_captions": result.alternative_captions,
                }
            except Exception as e:
                logger.warning(f"Captioning failed: {e}")

        # Fallback to scene description
        scene = await self.understand_scene(audio)
        return {
            "caption": scene.describe(),
            "confidence": 0.5,
            "alternative_captions": [],
        }

    # ========================
    # NICHE SOTA: Audio Quality Assessment
    # ========================

    async def assess_quality(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> dict[str, Any]:
        """
        Assess audio quality using NISQA/DNSMOS.

        Non-intrusive quality assessment without reference signal.
        Useful for:
        - Evaluating recording quality
        - Comparing enhancement algorithms
        - Quality control for audio pipelines

        Args:
            audio: Audio to assess

        Returns:
            Dictionary with:
            - overall_mos: Mean Opinion Score (1-5)
            - noisiness: Noise level score
            - coloration: Frequency distortion score
            - discontinuity: Temporal artifacts score
            - loudness: Volume appropriateness
            - speech_quality: Speech-specific MOS (if applicable)
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        if self._sota_status["quality_assessment"] and self._quality_assessor:
            try:
                result = await self._quality_assessor.assess(audio)
                self._stats["quality_assessments"] += 1

                logger.debug(
                    "Quality assessed",
                    mos=result.overall_mos,
                    model=result.model_used,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

                return {
                    "overall_mos": result.overall_mos,
                    "noisiness": result.noisiness,
                    "coloration": result.coloration,
                    "discontinuity": result.discontinuity,
                    "loudness": result.loudness,
                    "speech_quality": result.speech_quality,
                    "background_noise_level": result.background_noise_level,
                    "model": result.model_used,
                }
            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")

        # Basic fallback
        return {
            "overall_mos": 3.0,
            "noisiness": 3.0,
            "coloration": 3.0,
            "discontinuity": 3.0,
            "loudness": 3.0,
            "model": "fallback",
            "note": "Quality assessment not available. Install NISQA for accurate metrics.",
        }

    async def compare_audio_quality(
        self,
        original: Union[str, Path, np.ndarray, AudioSegment],
        processed: Union[str, Path, np.ndarray, AudioSegment],
    ) -> dict[str, Any]:
        """
        Compare quality between original and processed audio.

        Useful for evaluating enhancement, noise reduction, or compression.

        Args:
            original: Original/reference audio
            processed: Processed/enhanced audio

        Returns:
            Dictionary with quality comparison metrics
        """
        if not self._initialized:
            await self.initialize()

        if self._sota_status["quality_assessment"] and self._quality_assessor:
            return await self._quality_assessor.compare_quality(original, processed)

        # Fallback: assess both independently
        original_quality = await self.assess_quality(original)
        processed_quality = await self.assess_quality(processed)

        return {
            "original_mos": original_quality["overall_mos"],
            "processed_mos": processed_quality["overall_mos"],
            "mos_improvement": processed_quality["overall_mos"] - original_quality["overall_mos"],
            "original_details": original_quality,
            "processed_details": processed_quality,
        }

    # ========================
    # NICHE SOTA: Sound Event Localization and Detection (SELD)
    # ========================

    async def detect_and_localize_events(
        self,
        audio: Union[str, Path, np.ndarray],
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """
        Detect sound events and estimate their spatial locations.

        Sound Event Localization and Detection (SELD) combines:
        - WHAT sounds are present (event detection)
        - WHERE sounds are coming from (spatial localization)

        Supports:
        - Mono: Detection only
        - Stereo: Basic left/right localization
        - Ambisonics: Full 3D localization (azimuth + elevation)
        - Multi-channel: Precise spatial mapping

        Args:
            audio: Audio input (mono, stereo, or multi-channel)
            threshold: Detection confidence threshold

        Returns:
            Dictionary with:
            - events: List of localized events with spatial coordinates
            - spatial_resolution: "mono", "stereo", "ambisonics", "multi-channel"
            - duration: Audio duration
        """
        if not self._initialized:
            await self.initialize()

        if not self._sota_status["seld"] or not self._seld:
            raise RuntimeError(
                "SELD not available. Enable with use_seld=True in config."
            )

        start_time = time.time()

        result = await self._seld.detect_and_localize(
            audio,
            threshold=threshold,
            frame_duration=self.config.seld_frame_duration,
        )
        self._stats["seld_detections"] += 1

        logger.debug(
            "SELD detection complete",
            num_events=len(result.events),
            spatial_resolution=result.spatial_resolution,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        return {
            "events": [
                {
                    "label": e.label,
                    "confidence": e.confidence,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "azimuth": e.azimuth,
                    "elevation": e.elevation,
                    "distance": e.distance,
                }
                for e in result.events
            ],
            "num_channels": result.num_channels,
            "duration": result.duration,
            "spatial_resolution": result.spatial_resolution,
        }

    # ========================
    # NICHE SOTA: Audio Inpainting (AudioLDM2)
    # ========================

    async def inpaint_audio(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        mask_start: float,
        mask_end: float,
        prompt: Optional[str] = None,
    ) -> AudioSegment:
        """
        Inpaint (fill in) a section of audio using AudioLDM2.

        Uses diffusion-based generation to seamlessly fill missing
        or corrupted sections of audio.

        Args:
            audio: Original audio with section to replace
            mask_start: Start of section to inpaint (seconds)
            mask_end: End of section to inpaint (seconds)
            prompt: Optional text prompt to guide generation style

        Returns:
            AudioSegment with inpainted section
        """
        if not self._initialized:
            await self.initialize()

        if not self._sota_status["inpainting"] or not self._inpainter:
            raise RuntimeError(
                "Audio inpainting not available. Enable with use_inpainting=True "
                "and install diffusers + AudioLDM2."
            )

        start_time = time.time()

        result = await self._inpainter.inpaint(
            audio,
            mask_start=mask_start,
            mask_end=mask_end,
            prompt=prompt,
            num_inference_steps=self.config.inpainting_steps,
        )
        self._stats["inpaintings"] += 1

        logger.debug(
            "Audio inpainted",
            mask_start=mask_start,
            mask_end=mask_end,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        return AudioSegment(
            waveform=result.waveform,
            sample_rate=result.sample_rate,
            metadata={
                "inpainted": True,
                "mask_start": result.mask_start,
                "mask_end": result.mask_end,
                "prompt": result.prompt,
            },
        )

    async def continue_audio(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        continuation_duration: float = 5.0,
        prompt: Optional[str] = None,
    ) -> AudioSegment:
        """
        Continue/extend audio using AudioLDM2.

        Generates natural continuation of the input audio.

        Args:
            audio: Original audio to continue
            continuation_duration: Duration to generate (seconds)
            prompt: Text prompt for continuation style

        Returns:
            AudioSegment with extended audio
        """
        if not self._sota_status["inpainting"] or not self._inpainter:
            raise RuntimeError(
                "Audio continuation requires AudioLDM2. Enable use_inpainting=True."
            )

        result = await self._inpainter.continue_audio(
            audio,
            continuation_duration=continuation_duration,
            prompt=prompt,
        )

        return AudioSegment(
            waveform=result.waveform,
            sample_rate=result.sample_rate,
            metadata={
                "continued": True,
                "original_end": result.mask_start,
                "continuation_duration": continuation_duration,
                "prompt": result.prompt,
            },
        )

    async def generate_audio(
        self,
        prompt: str,
        duration: float = 10.0,
    ) -> AudioSegment:
        """
        Generate audio from text description using AudioLDM2.

        Args:
            prompt: Text description of desired audio
                   (e.g., "rain falling on a tin roof with distant thunder")
            duration: Duration in seconds

        Returns:
            AudioSegment with generated audio
        """
        if not self._sota_status["inpainting"] or not self._inpainter:
            raise RuntimeError(
                "Audio generation requires AudioLDM2. Enable use_inpainting=True."
            )

        result = await self._inpainter.generate(
            prompt=prompt,
            duration=duration,
            num_inference_steps=self.config.inpainting_steps,
        )

        return AudioSegment(
            waveform=result.waveform,
            sample_rate=result.sample_rate,
            metadata={
                "generated": True,
                "prompt": prompt,
                "duration": duration,
            },
        )

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
                "available": self._sota_status["emotion"] or self._sota_status["emotion2vec"],
                "sota": self._sota_status["emotion2vec"],  # emotion2vec is TRUE SOTA
                "features": {
                    "dimensional": self._sota_status["emotion"],
                    "categorical": True,
                    "emotion2vec": self._sota_status["emotion2vec"],  # TRUE SOTA
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
            # TRUE SOTA additions
            "event_detection": {
                "available": True,
                "sota": self._sota_status["beats"],  # BEATs is TRUE SOTA
                "features": {
                    "beats": self._sota_status["beats"],
                    "audioset_labels": True,
                },
            },
            "speech_enhancement": {
                "available": self._sota_status["deepfilter"],
                "sota": self._sota_status["deepfilter"],
                "features": {
                    "noise_reduction": self._sota_status["deepfilter"],
                    "real_time": self._sota_status["deepfilter"],
                },
            },
            "streaming_asr": {
                "available": self._sota_status["streaming_asr"],
                "sota": True,  # faster-whisper is SOTA for streaming
                "features": {
                    "real_time": self._sota_status["streaming_asr"],
                    "word_timestamps": self._sota_status["streaming_asr"],
                    "vad": self._sota_status["streaming_asr"],
                },
            },
            "music_generation": {
                "available": self._sota_status["musicgen"],
                "sota": self._sota_status["musicgen"],  # MusicGen is SOTA
                "features": {
                    "text_to_music": self._sota_status["musicgen"],
                    "melody_conditioning": self.config.musicgen_model == "melody",
                },
            },
            "audio_captioning": {
                "available": self._sota_status["captioner"],
                "sota": self._sota_status["captioner"],
                "features": {
                    "clap_based": self._sota_status["captioner"],
                },
            },
            # Niche SOTA additions
            "quality_assessment": {
                "available": self._sota_status["quality_assessment"],
                "sota": self._sota_status["quality_assessment"],
                "features": {
                    "nisqa": self._sota_status["quality_assessment"],
                    "dnsmos": self._sota_status["quality_assessment"],
                    "non_intrusive": True,  # No reference required
                    "mos_prediction": self._sota_status["quality_assessment"],
                },
            },
            "spatial_audio": {
                "available": self._sota_status["seld"],
                "sota": self._sota_status["seld"],
                "features": {
                    "event_localization": self._sota_status["seld"],
                    "azimuth_estimation": self._sota_status["seld"],
                    "elevation_estimation": self._sota_status["seld"],
                    "stereo_support": self._sota_status["seld"],
                    "ambisonics_support": self._sota_status["seld"],
                },
            },
            "audio_inpainting": {
                "available": self._sota_status["inpainting"],
                "sota": self._sota_status["inpainting"],
                "features": {
                    "inpainting": self._sota_status["inpainting"],
                    "continuation": self._sota_status["inpainting"],
                    "text_to_audio": self._sota_status["inpainting"],
                    "audioldm2": self._sota_status["inpainting"],
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
