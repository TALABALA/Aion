"""
AION Auditory Cortex System

Complete audio perception and understanding providing:
- Speech recognition with speaker diarization (Whisper + pyannote)
- Audio event detection and scene understanding (AST, BEATs)
- Speaker identification and verification (ECAPA-TDNN)
- Text-to-speech generation (Bark, TTS)
- Audio memory and retrieval (CLAP + FAISS)
- Music analysis (librosa, essentia)
- LLM-powered reasoning and QA

Example usage:
    ```python
    from aion.systems.audio import AuditoryCortex

    # Initialize
    cortex = AuditoryCortex()
    await cortex.initialize()

    # Transcribe with speaker diarization
    transcript = await cortex.transcribe("meeting.wav", enable_diarization=True)

    # Full scene understanding
    scene = await cortex.understand_scene("audio.wav")

    # Detect audio events
    events = await cortex.detect_events("scene.wav")

    # Generate speech
    audio = await cortex.synthesize("Hello, world!")

    # Store in memory
    await cortex.remember("call.wav", context="Client meeting")

    # Recall similar
    memories = await cortex.recall_similar("revenue discussion")

    # Answer questions about audio
    answer = await cortex.answer_question("podcast.mp3", "What topics were discussed?")

    # Clean up
    await cortex.shutdown()
    ```
"""

from aion.systems.audio.models import (
    # Enums
    AudioFormat,
    AudioEventType,
    EmotionalTone,
    SpeechStyle,
    # Core data structures
    TimeRange,
    AudioSegment,
    Word,
    TranscriptSegment,
    Speaker,
    Transcript,
    AudioEvent,
    MusicAnalysis,
    AudioQuality,
    AudioScene,
    VoiceCharacteristics,
    # Synthesis
    SynthesisRequest,
    SynthesisResult,
    # Memory
    AudioMemoryEntry,
    # Comparison and reasoning
    AudioComparisonResult,
    AudioReasoningResult,
)

from aion.systems.audio.perception import (
    AudioPerception,
    PerceptionConfig,
)

from aion.systems.audio.memory import (
    AudioMemory,
    AudioSearchResult,
    Conversation,
    ConversationSegment,
)

from aion.systems.audio.cortex import (
    AuditoryCortex,
    AuditoryCortexConfig,
    AnalysisResult,
)

from aion.systems.audio.sota_audio import (
    # Chain-of-Thought
    AudioThought,
    AudioCoTResult,
    AudioCoTReasoner,
    # Audio-Visual Fusion
    AudioVisualCorrelation,
    AudioVisualScene,
    AudioVisualFusion,
    # Conversation Tracking
    ConversationTurn,
    ConversationSummary,
    ConversationTracker,
    # Emotion Analysis
    EmotionAnalysisResult,
    EmotionAnalyzer,
    # Combined SOTA
    SOTAAudioCortex,
)

from aion.systems.audio.reasoning import (
    ReasoningContext,
    AudioReasoner,
    MultiModalReasoner,
)

# SOTA model integrations (optional, require additional dependencies)
try:
    from aion.systems.audio.sota_models import (
        # Whisper-X
        WhisperXConfig,
        WhisperXTranscriber,
        # Emotion Recognition (wav2vec2-based)
        EmotionResult,
        SpeechEmotionRecognizer,
        # TRUE SOTA Emotion (emotion2vec)
        Emotion2VecResult,
        Emotion2VecRecognizer,
        # Source Separation
        SeparatedSources,
        AudioSourceSeparator,
        # Advanced TTS
        XTTSConfig,
        XTTSSynthesizer,
        # Audio LLM
        AudioLanguageModel,
        # Combined Engine
        SOTAAudioEngine,
        # TRUE SOTA Audio Events (BEATs)
        AudioEventResult,
        BEATsEventDetector,
        # Speech Enhancement
        EnhancedAudio,
        DeepFilterNetEnhancer,
        # Streaming ASR
        StreamingTranscript,
        StreamingASR,
        # Music Generation
        GeneratedMusic,
        MusicGenerator,
        # Audio Captioning
        AudioCaption,
        AudioCaptioner,
        # Audio Quality Assessment
        AudioQualityResult,
        AudioQualityAssessor,
        # Sound Event Localization and Detection (SELD)
        LocalizedEvent,
        SELDResult,
        SELDDetector,
        # Audio Inpainting
        InpaintedAudio,
        AudioInpainter,
    )
    _SOTA_AVAILABLE = True
except ImportError:
    _SOTA_AVAILABLE = False

__all__ = [
    # Main interface
    "AuditoryCortex",
    "AuditoryCortexConfig",
    "AnalysisResult",
    # Enums
    "AudioFormat",
    "AudioEventType",
    "EmotionalTone",
    "SpeechStyle",
    # Core data structures
    "TimeRange",
    "AudioSegment",
    "Word",
    "TranscriptSegment",
    "Speaker",
    "Transcript",
    "AudioEvent",
    "MusicAnalysis",
    "AudioQuality",
    "AudioScene",
    "VoiceCharacteristics",
    # Synthesis
    "SynthesisRequest",
    "SynthesisResult",
    # Memory
    "AudioMemoryEntry",
    "AudioMemory",
    "AudioSearchResult",
    "Conversation",
    "ConversationSegment",
    # Comparison and reasoning
    "AudioComparisonResult",
    "AudioReasoningResult",
    # Perception
    "AudioPerception",
    "PerceptionConfig",
    # SOTA features
    "AudioThought",
    "AudioCoTResult",
    "AudioCoTReasoner",
    "AudioVisualCorrelation",
    "AudioVisualScene",
    "AudioVisualFusion",
    "ConversationTurn",
    "ConversationSummary",
    "ConversationTracker",
    "EmotionAnalysisResult",
    "EmotionAnalyzer",
    "SOTAAudioCortex",
    # Reasoning
    "ReasoningContext",
    "AudioReasoner",
    "MultiModalReasoner",
    # SOTA models (conditionally available)
    "WhisperXConfig",
    "WhisperXTranscriber",
    "EmotionResult",
    "SpeechEmotionRecognizer",
    "SeparatedSources",
    "AudioSourceSeparator",
    "XTTSConfig",
    "XTTSSynthesizer",
    "AudioLanguageModel",
    "SOTAAudioEngine",
    # TRUE SOTA additions
    "Emotion2VecResult",
    "Emotion2VecRecognizer",
    "AudioEventResult",
    "BEATsEventDetector",
    "EnhancedAudio",
    "DeepFilterNetEnhancer",
    "StreamingTranscript",
    "StreamingASR",
    "GeneratedMusic",
    "MusicGenerator",
    "AudioCaption",
    "AudioCaptioner",
    # Niche SOTA additions
    "AudioQualityResult",
    "AudioQualityAssessor",
    "LocalizedEvent",
    "SELDResult",
    "SELDDetector",
    "InpaintedAudio",
    "AudioInpainter",
]
