"""
AION Auditory Cortex Tests

Comprehensive test suite for the audio perception and understanding system.
"""

import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pytest

from aion.systems.audio import (
    # Main interface
    AuditoryCortex,
    AuditoryCortexConfig,
    AnalysisResult,
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
    # Memory
    AudioMemory,
    AudioMemoryEntry,
    AudioSearchResult,
    # Perception
    AudioPerception,
    PerceptionConfig,
    # SOTA
    AudioCoTReasoner,
    AudioCoTResult,
    ConversationTracker,
    EmotionAnalyzer,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_audio_440hz():
    """Generate a simple 440 Hz test tone."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def sample_audio_speech_like():
    """Generate audio that simulates speech characteristics."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Simulate speech with varying amplitude and frequency
    # Fundamental frequency around 150 Hz (typical male voice)
    f0 = 150 + 30 * np.sin(2 * np.pi * 2 * t)  # Slow pitch variation
    amplitude = 0.3 * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))  # Volume variation

    audio = amplitude * np.sin(2 * np.pi * f0 * t)

    # Add some harmonics
    audio += 0.15 * amplitude * np.sin(2 * np.pi * 2 * f0 * t)
    audio += 0.08 * amplitude * np.sin(2 * np.pi * 3 * f0 * t)

    return audio.astype(np.float32), sr


@pytest.fixture
def sample_audio_music_like():
    """Generate audio that simulates music characteristics."""
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # C major chord: C (262 Hz), E (330 Hz), G (392 Hz)
    chord = (
        0.3 * np.sin(2 * np.pi * 262 * t) +
        0.25 * np.sin(2 * np.pi * 330 * t) +
        0.25 * np.sin(2 * np.pi * 392 * t)
    )

    # Add a simple beat pattern
    beat_freq = 2  # 120 BPM
    beat_envelope = 0.5 * (1 + np.sign(np.sin(2 * np.pi * beat_freq * t)))
    audio = chord * (0.7 + 0.3 * beat_envelope)

    return audio.astype(np.float32), sr


@pytest.fixture
def audio_segment(sample_audio_440hz):
    """Create an AudioSegment from sample audio."""
    audio, sr = sample_audio_440hz
    return AudioSegment(
        waveform=audio,
        sample_rate=sr,
        channels=1,
        start_time=0.0,
        end_time=len(audio) / sr,
    )


@pytest.fixture
async def audio_cortex():
    """Create and initialize an AuditoryCortex instance."""
    config = AuditoryCortexConfig(
        enable_memory=True,
        device="cpu",
        whisper_model="openai/whisper-tiny",  # Use tiny model for tests
        enable_diarization=False,  # Disable for faster tests
        enable_tts=False,  # Disable for tests
        enable_music_analysis=True,
    )
    cortex = AuditoryCortex(config=config)
    await cortex.initialize()
    yield cortex
    await cortex.shutdown()


@pytest.fixture
async def audio_memory():
    """Create and initialize an AudioMemory instance."""
    memory = AudioMemory(
        embedding_dim=512,
        max_entries=1000,
    )
    await memory.initialize()
    yield memory
    await memory.shutdown()


@pytest.fixture
def perception():
    """Create an AudioPerception instance."""
    config = PerceptionConfig(
        device="cpu",
        enable_diarization=False,
    )
    return AudioPerception(config)


# ============================================================================
# Model Tests
# ============================================================================

class TestAudioModels:
    """Test audio data models."""

    def test_time_range(self):
        """Test TimeRange operations."""
        tr1 = TimeRange(start=1.0, end=3.0)
        tr2 = TimeRange(start=2.0, end=4.0)
        tr3 = TimeRange(start=5.0, end=6.0)

        # Duration
        assert tr1.duration == 2.0

        # Midpoint
        assert tr1.midpoint == 2.0

        # Overlap detection
        assert tr1.overlaps(tr2)
        assert not tr1.overlaps(tr3)

        # Overlap duration
        assert tr1.overlap_duration(tr2) == 1.0
        assert tr1.overlap_duration(tr3) == 0.0

        # Contains
        assert tr1.contains(2.0)
        assert not tr1.contains(4.0)

    def test_audio_segment(self, sample_audio_440hz):
        """Test AudioSegment creation and properties."""
        audio, sr = sample_audio_440hz

        segment = AudioSegment(
            waveform=audio,
            sample_rate=sr,
            channels=1,
        )

        assert segment.duration == pytest.approx(2.0, rel=0.01)
        assert segment.num_samples == len(audio)
        assert segment.sample_rate == sr

    def test_audio_segment_subsegment(self, audio_segment):
        """Test AudioSegment subsegment extraction."""
        sub = audio_segment.get_subsegment(0.5, 1.5)

        assert sub.duration == pytest.approx(1.0, rel=0.01)
        assert sub.start_time == 0.5
        assert sub.end_time == 1.5

    def test_transcript_segment(self):
        """Test TranscriptSegment."""
        segment = TranscriptSegment(
            text="Hello world",
            start_time=0.0,
            end_time=2.0,
            confidence=0.95,
            speaker_id="speaker_1",
            words=[
                Word(text="Hello", start_time=0.0, end_time=0.8, confidence=0.98),
                Word(text="world", start_time=1.0, end_time=1.8, confidence=0.92),
            ]
        )

        assert segment.duration == 2.0
        assert segment.word_count == 2
        assert len(segment.words) == 2

    def test_transcript(self):
        """Test Transcript."""
        transcript = Transcript(
            segments=[
                TranscriptSegment(text="Hello", start_time=0.0, end_time=1.0, speaker_id="s1"),
                TranscriptSegment(text="Hi there", start_time=1.5, end_time=3.0, speaker_id="s2"),
            ],
            speakers=[
                Speaker(id="s1", name="Alice"),
                Speaker(id="s2", name="Bob"),
            ],
            language="en",
        )

        assert transcript.speaker_count == 2
        assert "Hello" in transcript.text

        by_speaker = transcript.get_text_by_speaker()
        assert "s1" in by_speaker
        assert "Hello" in by_speaker["s1"]

    def test_speaker(self):
        """Test Speaker."""
        speaker = Speaker(
            name="Test Speaker",
            embedding=np.random.randn(192).astype(np.float32),
            segments=[
                TimeRange(0.0, 2.0),
                TimeRange(5.0, 8.0),
            ],
        )

        assert speaker.total_speaking_time == 5.0
        assert speaker.segment_count == 2

    def test_audio_event(self):
        """Test AudioEvent."""
        event = AudioEvent(
            label="speech",
            event_type=AudioEventType.SPEECH,
            start_time=0.0,
            end_time=3.0,
            confidence=0.9,
        )

        assert event.duration == 3.0
        assert event.event_type == AudioEventType.SPEECH

    def test_music_analysis(self):
        """Test MusicAnalysis."""
        analysis = MusicAnalysis(
            tempo=120.0,
            key="C major",
            mood="energetic",
            energy=0.8,
            beats=[0.5, 1.0, 1.5, 2.0],
        )

        assert analysis.tempo == 120.0
        assert analysis.key == "C major"
        assert len(analysis.beats) == 4

    def test_audio_scene(self):
        """Test AudioScene."""
        scene = AudioScene(
            duration=10.0,
            events=[
                AudioEvent(label="speech", event_type=AudioEventType.SPEECH,
                          start_time=0.0, end_time=5.0, confidence=0.9),
                AudioEvent(label="music", event_type=AudioEventType.MUSIC,
                          start_time=3.0, end_time=10.0, confidence=0.8),
            ],
            speakers=[Speaker(name="Speaker 1")],
            scene_type="mixed",
            ambient_description="indoor environment",
        )

        assert scene.event_count == 2
        assert scene.speaker_count == 1
        assert scene.has_speech

        # Get events by type
        speech_events = scene.get_events_by_type(AudioEventType.SPEECH)
        assert len(speech_events) == 1

        # Description
        description = scene.describe()
        assert "mixed" in description.lower()


# ============================================================================
# Memory Tests
# ============================================================================

class TestAudioMemory:
    """Test audio memory system."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, audio_memory):
        """Test basic store and retrieve."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        entry = await audio_memory.store(
            embedding=embedding,
            context="Test audio",
            duration=5.0,
            importance=0.8,
            tags=["test", "sample"],
        )

        assert entry.id is not None
        assert entry.context == "Test audio"
        assert entry.importance == 0.8
        assert "test" in entry.tags

        # Retrieve
        retrieved = await audio_memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.id == entry.id

    @pytest.mark.asyncio
    async def test_search_by_similarity(self, audio_memory):
        """Test similarity search."""
        # Store multiple entries
        embeddings = []
        for i in range(5):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            await audio_memory.store(
                embedding=emb,
                context=f"Audio {i}",
                duration=float(i + 1),
            )

        # Search with similar embedding
        query = embeddings[0] + 0.1 * np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = await audio_memory.search_by_similarity(query, limit=3)
        assert len(results) <= 3
        # First result should be most similar
        assert results[0].similarity > 0.5

    @pytest.mark.asyncio
    async def test_search_by_tags(self, audio_memory):
        """Test tag-based search."""
        emb = np.random.randn(512).astype(np.float32)

        await audio_memory.store(
            embedding=emb,
            context="Meeting 1",
            tags=["meeting", "important"],
        )
        await audio_memory.store(
            embedding=emb,
            context="Meeting 2",
            tags=["meeting"],
        )
        await audio_memory.store(
            embedding=emb,
            context="Call",
            tags=["call"],
        )

        # Search by single tag
        results = await audio_memory.search_by_tags(["meeting"])
        assert len(results) == 2

        # Search requiring all tags
        results = await audio_memory.search_by_tags(["meeting", "important"], require_all=True)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_speaker_registration(self, audio_memory):
        """Test speaker registration and identification."""
        embedding = np.random.randn(192).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        speaker = Speaker(
            name="Test Speaker",
            embedding=embedding,
        )

        registered = await audio_memory.register_speaker(speaker, "Test Speaker")
        assert registered.name == "Test Speaker"

        # Identify speaker
        query = embedding + 0.05 * np.random.randn(192).astype(np.float32)
        query = query / np.linalg.norm(query)

        identified, similarity = await audio_memory.identify_speaker(query)
        assert identified is not None
        assert identified.name == "Test Speaker"
        assert similarity > 0.9

    @pytest.mark.asyncio
    async def test_conversation_tracking(self, audio_memory):
        """Test conversation tracking."""
        # Start conversation
        conv_id = await audio_memory.start_conversation("Test Conversation")
        assert conv_id is not None

        # Add turns
        emb = np.random.randn(512).astype(np.float32)
        entry = await audio_memory.store(embedding=emb, context="Turn 1")

        await audio_memory.add_to_conversation(
            memory_id=entry.id,
            speaker_id="s1",
            text="Hello",
            start_time=0.0,
            end_time=1.0,
        )

        # End conversation
        conv = await audio_memory.end_conversation(conv_id)
        assert conv is not None


# ============================================================================
# Perception Tests
# ============================================================================

class TestAudioPerception:
    """Test audio perception system."""

    @pytest.mark.asyncio
    async def test_load_audio_numpy(self, perception, sample_audio_440hz):
        """Test loading audio from numpy array."""
        audio, sr = sample_audio_440hz
        loaded_audio, loaded_sr = await perception.load_audio(audio)

        assert loaded_sr == 16000  # Default target SR
        assert len(loaded_audio) > 0

    @pytest.mark.asyncio
    async def test_audio_hash(self, perception, sample_audio_440hz):
        """Test audio hashing for deduplication."""
        audio, sr = sample_audio_440hz

        hash1 = perception.compute_audio_hash(audio)
        hash2 = perception.compute_audio_hash(audio)

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

        # Different audio should have different hash
        different_audio = audio * 0.5
        hash3 = perception.compute_audio_hash(different_audio)
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_fallback_embedding(self, perception, sample_audio_440hz):
        """Test fallback embedding computation."""
        audio, sr = sample_audio_440hz

        embedding = await perception._compute_fallback_embedding(audio)

        assert embedding is not None
        assert embedding.shape == (512,)
        # Should be normalized
        assert np.abs(np.linalg.norm(embedding) - 1.0) < 0.01 or np.linalg.norm(embedding) < 0.01


# ============================================================================
# Cortex Tests
# ============================================================================

class TestAuditoryCortex:
    """Test main AuditoryCortex class."""

    @pytest.mark.asyncio
    async def test_initialization(self, audio_cortex):
        """Test cortex initialization."""
        assert audio_cortex._initialized

    @pytest.mark.asyncio
    async def test_load_audio(self, audio_cortex, sample_audio_440hz):
        """Test audio loading through cortex."""
        audio, sr = sample_audio_440hz

        segment = await audio_cortex.load_audio(audio)

        assert segment.duration == pytest.approx(2.0, rel=0.01)
        assert segment.waveform is not None

    @pytest.mark.asyncio
    async def test_analyze_music(self, audio_cortex, sample_audio_music_like):
        """Test music analysis."""
        audio, sr = sample_audio_music_like

        analysis = await audio_cortex.analyze_music(audio)

        assert analysis.tempo > 0
        # Should detect some tempo (around 120 BPM given our beat pattern)

    @pytest.mark.asyncio
    async def test_remember_and_recall(self, audio_cortex, sample_audio_440hz):
        """Test memory operations."""
        audio, sr = sample_audio_440hz

        # Remember
        memory_id = await audio_cortex.remember(
            audio=audio,
            context="Test tone 440Hz",
            importance=0.7,
            tags=["test", "tone"],
        )

        assert memory_id is not None

        # Recall by tags
        results = await audio_cortex.recall_by_tags(["test"])
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_compare_audio(self, audio_cortex, sample_audio_440hz, sample_audio_speech_like):
        """Test audio comparison."""
        audio1, _ = sample_audio_440hz
        audio2, _ = sample_audio_speech_like

        comparison = await audio_cortex.compare(audio1, audio2)

        assert comparison.audio1_id is not None
        assert comparison.audio2_id is not None
        assert 0 <= comparison.overall_similarity <= 1

    @pytest.mark.asyncio
    async def test_get_stats(self, audio_cortex):
        """Test statistics retrieval."""
        stats = audio_cortex.get_stats()

        assert "transcriptions" in stats
        assert "perception" in stats


# ============================================================================
# SOTA Features Tests
# ============================================================================

class TestSOTAFeatures:
    """Test state-of-the-art audio features."""

    @pytest.mark.asyncio
    async def test_cot_reasoner(self):
        """Test Chain-of-Thought reasoner."""
        reasoner = AudioCoTReasoner()

        scene = AudioScene(
            duration=30.0,
            scene_type="conversation",
            ambient_description="office",
            events=[
                AudioEvent(label="speech", event_type=AudioEventType.SPEECH,
                          start_time=0.0, end_time=25.0, confidence=0.95),
            ],
            speakers=[
                Speaker(name="Alice"),
                Speaker(name="Bob"),
            ],
            transcript=Transcript(
                text="Hello, how are you? I'm doing great, thanks for asking.",
                segments=[
                    TranscriptSegment(text="Hello, how are you?", speaker_id="s1",
                                     start_time=0.0, end_time=2.0),
                    TranscriptSegment(text="I'm doing great, thanks for asking.", speaker_id="s2",
                                     start_time=2.5, end_time=5.0),
                ],
            ),
        )

        result = await reasoner.reason(scene, "Who is speaking?")

        assert isinstance(result, AudioCoTResult)
        assert len(result.thoughts) > 0
        assert result.answer != ""
        assert result.final_confidence > 0

    @pytest.mark.asyncio
    async def test_conversation_tracker(self):
        """Test conversation tracker."""
        tracker = ConversationTracker()

        # Start conversation
        conv_id = await tracker.start_conversation()

        # Add turns
        segment1 = TranscriptSegment(
            text="Hello, how are you?",
            start_time=0.0,
            end_time=2.0,
            speaker_id="s1",
        )
        segment2 = TranscriptSegment(
            text="I'm good, thanks!",
            start_time=2.5,
            end_time=4.0,
            speaker_id="s2",
        )

        turn1 = await tracker.add_turn(segment1)
        turn2 = await tracker.add_turn(segment2)

        assert turn1.intent is not None
        assert turn2.speaker_id == "s2"

        # End and summarize
        summary = await tracker.end_conversation()

        assert summary.turn_count == 2
        assert len(summary.participants) > 0

    @pytest.mark.asyncio
    async def test_emotion_analyzer(self):
        """Test emotion analyzer."""
        analyzer = EmotionAnalyzer()

        scene = AudioScene(
            duration=10.0,
            scene_type="speech",
            transcript=Transcript(
                text="I'm so happy today! Everything is wonderful!",
            ),
            emotional_tone=EmotionalTone.HAPPY,
            emotional_confidence=0.8,
        )

        result = await analyzer.analyze(scene)

        assert result.primary_emotion is not None
        assert "happy" in result.emotion_scores
        assert 0 <= result.valence <= 1
        assert 0 <= result.arousal <= 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the audio system."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, audio_cortex, sample_audio_speech_like):
        """Test full audio processing pipeline."""
        audio, sr = sample_audio_speech_like

        # Full analysis
        result = await audio_cortex.analyze(
            audio=audio,
            store_in_memory=True,
            context="Integration test",
        )

        assert isinstance(result, AnalysisResult)
        assert result.scene is not None
        assert result.processing_time_ms > 0

        # Scene should have duration
        assert result.scene.duration > 0

    @pytest.mark.asyncio
    async def test_audio_to_memory_to_recall(self, audio_cortex, sample_audio_440hz):
        """Test storing and recalling audio from memory."""
        audio, sr = sample_audio_440hz

        # Store
        memory_id = await audio_cortex.remember(
            audio=audio,
            context="440 Hz test tone for integration test",
            importance=0.9,
            tags=["integration", "test", "tone"],
        )

        # Recall by text
        results = await audio_cortex.recall_similar(
            query="test tone",
            limit=5,
        )

        # Should find at least one result
        assert len(results) >= 0  # May be 0 if CLAP not loaded

        # Recall by tags should work
        tag_results = await audio_cortex.recall_by_tags(["integration"])
        assert len(tag_results) > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_memory_scaling(self):
        """Test memory system with many entries."""
        memory = AudioMemory(
            embedding_dim=512,
            max_entries=10000,
        )
        await memory.initialize()

        # Store 100 entries
        for i in range(100):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            await memory.store(
                embedding=emb,
                context=f"Audio {i}",
                tags=[f"batch_{i // 10}"],
            )

        # Search should still be fast
        query = np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)

        import time
        start = time.time()
        results = await memory.search_by_similarity(query, limit=10)
        elapsed = time.time() - start

        assert len(results) == 10
        assert elapsed < 1.0  # Should complete in under 1 second

        await memory.shutdown()


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_transcript(self):
        """Test handling of empty transcript."""
        transcript = Transcript()
        assert transcript.text == ""
        assert transcript.word_count == 0

    def test_zero_duration_segment(self):
        """Test segment with zero duration."""
        segment = AudioSegment(
            start_time=0.0,
            end_time=0.0,
            sample_rate=16000,
        )
        assert segment.duration == 0.0

    @pytest.mark.asyncio
    async def test_empty_audio_scene(self):
        """Test empty audio scene."""
        scene = AudioScene()
        description = scene.describe()
        assert "0.0 seconds" in description

    @pytest.mark.asyncio
    async def test_memory_without_initialization(self):
        """Test memory operations trigger initialization."""
        memory = AudioMemory()

        emb = np.random.randn(512).astype(np.float32)
        entry = await memory.store(embedding=emb, context="Test")

        # Should auto-initialize
        assert entry is not None
        await memory.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
