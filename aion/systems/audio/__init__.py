"""AION Auditory Cortex - Audio perception, understanding, and reasoning."""

from aion.systems.audio.cortex import AuditoryCortex, AudioAttention
from aion.systems.audio.perception import AudioPerception
from aion.systems.audio.memory import AudioMemory
from aion.systems.audio.models import (
    AudioSegment,
    AudioScene,
    AudioEvent,
    AudioRelation,
    Transcript,
    TranscriptSegment,
    Word,
    Speaker,
    MusicAnalysis,
    TimeRange,
    FrequencyRange,
    AudioAnalysisResult,
    AudioMemoryEntry,
    AudioSearchResult,
    VoiceProfile,
)
from aion.systems.audio.sota_audio import (
    SOTAAudioCortex,
    AudioCoTReasoner,
    AudioVisualFusion,
    ConversationalMemory,
    StreamingAudioProcessor,
    SpeakerBehaviorAnalyzer,
)
from aion.systems.audio.reasoning import (
    AudioReasoner,
    AudioTranscriptAnalyzer,
    AudioReasoningEngine,
)

__all__ = [
    # Main cortex
    "AuditoryCortex",
    "AudioAttention",
    # Perception
    "AudioPerception",
    # Memory
    "AudioMemory",
    # Models
    "AudioSegment",
    "AudioScene",
    "AudioEvent",
    "AudioRelation",
    "Transcript",
    "TranscriptSegment",
    "Word",
    "Speaker",
    "MusicAnalysis",
    "TimeRange",
    "FrequencyRange",
    "AudioAnalysisResult",
    "AudioMemoryEntry",
    "AudioSearchResult",
    "VoiceProfile",
    # SOTA
    "SOTAAudioCortex",
    "AudioCoTReasoner",
    "AudioVisualFusion",
    "ConversationalMemory",
    "StreamingAudioProcessor",
    "SpeakerBehaviorAnalyzer",
    # Reasoning
    "AudioReasoner",
    "AudioTranscriptAnalyzer",
    "AudioReasoningEngine",
]
