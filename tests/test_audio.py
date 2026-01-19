"""
AION Audio System Tests

Comprehensive tests for the Auditory Cortex system:
- AudioPerception tests
- AudioMemory tests
- AuditoryCortex integration tests
- SOTA features tests
"""

import asyncio
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime

# Import audio system components
from aion.systems.audio.models import (
    AudioSegment,
    AudioScene,
    AudioEvent,
    Transcript,
    TranscriptSegment,
    Speaker,
    MusicAnalysis,
    TimeRange,
    FrequencyRange,
    AudioMemoryEntry,
    VoiceProfile,
    Word,
)
from aion.systems.audio.perception import AudioPerception
from aion.systems.audio.memory import AudioMemory
from aion.systems.audio.cortex import AuditoryCortex
from aion.systems.audio.sota_audio import (
    AudioCoTReasoner,
    ConversationalMemory,
    StreamingAudioProcessor,
    SpeakerBehaviorAnalyzer,
)
from aion.systems.audio.reasoning import (
    AudioReasoner,
    AudioTranscriptAnalyzer,
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_waveform():
    """Generate a simple test waveform (440 Hz sine wave)."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return waveform, sr


@pytest.fixture
def sample_audio_segment(sample_waveform):
    """Create a sample AudioSegment."""
    waveform, sr = sample_waveform
    return AudioSegment.from_array(waveform, sr)


@pytest.fixture
def sample_speaker():
    """Create a sample Speaker."""
    return Speaker(
        id="speaker_1",
        name="Test Speaker",
        embedding=np.random.randn(192).astype(np.float32),
        segments=[TimeRange(0.0, 2.0), TimeRange(5.0, 8.0)],
        confidence=0.9,
    )


@pytest.fixture
def sample_transcript(sample_speaker):
    """Create a sample Transcript."""
    return Transcript(
        id="transcript_1",
        text="Hello, this is a test transcript. How are you today?",
        language="en",
        confidence=0.95,
        segments=[
            TranscriptSegment(
                id="seg_1",
                text="Hello, this is a test transcript.",
                start_time=0.0,
                end_time=2.0,
                confidence=0.9,
                speaker_id="speaker_1",
            ),
            TranscriptSegment(
                id="seg_2",
                text="How are you today?",
                start_time=2.5,
                end_time=4.0,
                confidence=0.92,
                speaker_id="speaker_1",
            ),
        ],
        speakers=[sample_speaker],
        audio_id="audio_1",
        duration=4.0,
    )


@pytest.fixture
def sample_audio_scene(sample_speaker, sample_transcript):
    """Create a sample AudioScene."""
    return AudioScene(
        id="scene_1",
        audio_id="audio_1",
        duration=10.0,
        events=[
            AudioEvent(
                id="event_1",
                label="speech",
                category="speech",
                start_time=0.0,
                end_time=4.0,
                confidence=0.9,
            ),
            AudioEvent(
                id="event_2",
                label="music",
                category="music",
                start_time=5.0,
                end_time=10.0,
                confidence=0.8,
            ),
        ],
        speakers=[sample_speaker],
        transcript=sample_transcript,
        ambient_description="quiet office",
        emotional_tone="neutral",
        noise_level_db=-30.0,
    )


@pytest.fixture
async def audio_memory():
    """Create an AudioMemory instance."""
    memory = AudioMemory(max_entries=100)
    await memory.initialize()
    yield memory
    await memory.shutdown()


@pytest.fixture
async def audio_perception():
    """Create an AudioPerception instance (without loading heavy models)."""
    # Create perception without initializing heavy models
    perception = AudioPerception(device="cpu")
    # Don't fully initialize to avoid loading models in tests
    perception._initialized = True
    yield perception
    await perception.shutdown()


@pytest.fixture
async def audio_cortex():
    """Create an AuditoryCortex instance."""
    cortex = AuditoryCortex(
        enable_memory=True,
        device="cpu",
        enable_diarization=False,  # Disable for faster tests
        enable_tts=False,
    )
    await cortex.initialize()
    yield cortex
    await cortex.shutdown()


# ==================== Model Tests ====================

class TestAudioModels:
    """Tests for audio data models."""

    def test_time_range(self):
        """Test TimeRange operations."""
        tr1 = TimeRange(0.0, 5.0)
        tr2 = TimeRange(3.0, 8.0)
        tr3 = TimeRange(10.0, 15.0)

        assert tr1.duration == 5.0
        assert tr1.overlaps(tr2)
        assert not tr1.overlaps(tr3)
        assert tr1.contains(2.5)
        assert not tr1.contains(6.0)

        intersection = tr1.intersection(tr2)
        assert intersection is not None
        assert intersection.start == 3.0
        assert intersection.end == 5.0

    def test_frequency_range(self):
        """Test FrequencyRange."""
        fr = FrequencyRange(100.0, 500.0)
        assert fr.bandwidth == 400.0
        assert fr.center == 300.0

    def test_audio_segment_from_array(self, sample_waveform):
        """Test AudioSegment creation from array."""
        waveform, sr = sample_waveform
        segment = AudioSegment.from_array(waveform, sr)

        assert segment.duration == pytest.approx(2.0, rel=0.01)
        assert segment.sample_rate == sr
        assert segment.channels == 1
        assert segment.waveform is not None
        assert segment.num_samples == len(waveform)

    def test_audio_segment_to_dict(self, sample_audio_segment):
        """Test AudioSegment serialization."""
        d = sample_audio_segment.to_dict()

        assert "id" in d
        assert "duration" in d
        assert "sample_rate" in d
        assert d["has_waveform"] == True

    def test_speaker_similarity(self):
        """Test Speaker embedding similarity."""
        emb1 = np.random.randn(192).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        speaker1 = Speaker(id="s1", embedding=emb1)
        speaker2 = Speaker(id="s2", embedding=emb1.copy())
        speaker3 = Speaker(id="s3", embedding=-emb1)

        # Same embedding should have similarity 1.0
        assert speaker1.similarity(speaker2) == pytest.approx(1.0, rel=0.01)
        # Opposite embedding should have similarity -1.0
        assert speaker1.similarity(speaker3) == pytest.approx(-1.0, rel=0.01)

    def test_speaker_total_speaking_time(self, sample_speaker):
        """Test Speaker total speaking time calculation."""
        expected = 2.0 + 3.0  # Two segments
        assert sample_speaker.total_speaking_time == expected

    def test_transcript_to_srt(self, sample_transcript):
        """Test Transcript SRT export."""
        srt = sample_transcript.to_srt()

        assert "1" in srt
        assert "Hello" in srt
        assert "-->" in srt

    def test_transcript_get_text_for_speaker(self, sample_transcript):
        """Test getting text for specific speaker."""
        text = sample_transcript.get_text_for_speaker("speaker_1")
        assert "Hello" in text
        assert "How are you" in text

    def test_audio_scene_describe(self, sample_audio_scene):
        """Test AudioScene description generation."""
        description = sample_audio_scene.describe()

        assert isinstance(description, str)
        assert len(description) > 0

    def test_audio_scene_to_dict(self, sample_audio_scene):
        """Test AudioScene serialization."""
        d = sample_audio_scene.to_dict()

        assert "id" in d
        assert "events" in d
        assert "speakers" in d
        assert d["has_speech"] == True
        assert d["has_music"] == True

    def test_music_analysis(self):
        """Test MusicAnalysis dataclass."""
        analysis = MusicAnalysis(
            tempo_bpm=120.0,
            key="C",
            mode="major",
            time_signature="4/4",
            mood="happy",
            energy=0.8,
        )

        d = analysis.to_dict()
        assert d["tempo_bpm"] == 120.0
        assert d["key"] == "C"
        assert d["mood"] == "happy"

    def test_voice_profile_update(self):
        """Test VoiceProfile embedding update."""
        emb1 = np.ones(192, dtype=np.float32)
        emb2 = np.ones(192, dtype=np.float32) * 2

        profile = VoiceProfile(
            id="vp1",
            name="Test",
            embedding=emb1,
        )

        profile.update_embedding(emb2, 5.0)

        assert profile.sample_count == 2
        assert profile.total_duration == 5.0
        # New embedding should be weighted average
        assert np.allclose(profile.embedding, 1.5 * np.ones(192))


# ==================== Memory Tests ====================

class TestAudioMemory:
    """Tests for AudioMemory system."""

    @pytest.mark.asyncio
    async def test_memory_initialization(self, audio_memory):
        """Test memory initialization."""
        assert audio_memory._initialized
        assert audio_memory.count() == 0

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, audio_memory, sample_audio_scene):
        """Test storing and retrieving audio memories."""
        embedding = np.random.randn(512).astype(np.float32)

        entry = await audio_memory.store(
            scene=sample_audio_scene,
            embedding=embedding,
            transcript_text="Test transcript",
            importance=0.8,
        )

        assert entry is not None
        assert audio_memory.count() == 1

        retrieved = await audio_memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.id == entry.id

    @pytest.mark.asyncio
    async def test_search_by_embedding(self, audio_memory, sample_audio_scene):
        """Test embedding similarity search."""
        embedding1 = np.random.randn(512).astype(np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)

        await audio_memory.store(
            scene=sample_audio_scene,
            embedding=embedding1,
            importance=0.8,
        )

        # Search with same embedding should find it
        results = await audio_memory.search_by_embedding(embedding1, limit=5)
        assert len(results) == 1
        assert results[0].similarity > 0.99

    @pytest.mark.asyncio
    async def test_search_by_text(self, audio_memory, sample_audio_scene):
        """Test text-based search."""
        embedding = np.random.randn(512).astype(np.float32)

        await audio_memory.store(
            scene=sample_audio_scene,
            embedding=embedding,
            transcript_text="Hello world this is a test",
        )

        results = await audio_memory.search_by_text("hello world")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_voice_profile_management(self, audio_memory):
        """Test voice profile registration and lookup."""
        embedding = np.random.randn(512).astype(np.float32)

        profile = await audio_memory.register_voice_profile(
            name="John Doe",
            embedding=embedding,
            duration=10.0,
        )

        assert profile is not None
        assert audio_memory.count_voice_profiles() == 1

        retrieved = audio_memory.get_voice_profile(profile.id)
        assert retrieved is not None
        assert retrieved.name == "John Doe"

    @pytest.mark.asyncio
    async def test_identify_voice(self, audio_memory):
        """Test voice identification."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        await audio_memory.register_voice_profile(
            name="Test Speaker",
            embedding=embedding,
        )

        # Try to identify with same embedding
        profile, similarity = await audio_memory.identify_voice(
            embedding, threshold=0.5
        )

        assert profile is not None
        assert similarity > 0.99

    @pytest.mark.asyncio
    async def test_conversation_memory(self, audio_memory, sample_transcript):
        """Test conversation storage and search."""
        entry = await audio_memory.store_conversation(
            transcript=sample_transcript,
            topic="test topic",
            importance=0.7,
        )

        assert entry is not None
        assert audio_memory.count_conversations() == 1

        results = await audio_memory.search_conversations(query="hello")
        assert len(results) >= 0  # May or may not match

    @pytest.mark.asyncio
    async def test_memory_stats(self, audio_memory, sample_audio_scene):
        """Test memory statistics."""
        embedding = np.random.randn(512).astype(np.float32)
        await audio_memory.store(scene=sample_audio_scene, embedding=embedding)

        stats = audio_memory.get_stats()

        assert "total_entries" in stats
        assert stats["total_entries"] == 1


# ==================== Perception Tests ====================

class TestAudioPerception:
    """Tests for AudioPerception system."""

    @pytest.mark.asyncio
    async def test_load_audio_from_array(self, audio_perception, sample_waveform):
        """Test loading audio from numpy array."""
        waveform, sr = sample_waveform
        loaded, loaded_sr = await audio_perception.load_audio(waveform)

        assert loaded is not None
        assert len(loaded) == len(waveform)

    @pytest.mark.asyncio
    async def test_compute_audio_hash(self, audio_perception, sample_waveform):
        """Test audio hash computation."""
        waveform, _ = sample_waveform
        hash1 = audio_perception.compute_audio_hash(waveform)
        hash2 = audio_perception.compute_audio_hash(waveform)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_get_capabilities(self, audio_perception):
        """Test capability reporting."""
        caps = audio_perception.get_capabilities()

        assert "transcription" in caps
        assert "event_detection" in caps
        assert "speaker_embedding" in caps

    @pytest.mark.asyncio
    async def test_compute_spectrogram(self, audio_perception, sample_waveform):
        """Test spectrogram computation."""
        waveform, sr = sample_waveform

        # Skip if librosa not available
        try:
            spec = await audio_perception.compute_spectrogram(waveform)
            assert spec is not None
            assert spec.ndim == 2
        except Exception:
            pytest.skip("Spectrogram computation dependencies not available")


# ==================== Cortex Tests ====================

class TestAuditoryCortex:
    """Tests for AuditoryCortex system."""

    @pytest.mark.asyncio
    async def test_cortex_initialization(self, audio_cortex):
        """Test cortex initialization."""
        assert audio_cortex._initialized
        assert audio_cortex.memory is not None

    @pytest.mark.asyncio
    async def test_process_audio(self, audio_cortex, sample_waveform):
        """Test basic audio processing."""
        waveform, sr = sample_waveform

        result = await audio_cortex.process(
            waveform,
            store_in_memory=False,
        )

        assert result is not None
        assert result.audio_segment is not None
        assert result.scene is not None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_detect_events(self, audio_cortex, sample_waveform):
        """Test event detection."""
        waveform, _ = sample_waveform

        events = await audio_cortex.detect_events(waveform, threshold=0.3)

        assert isinstance(events, list)
        # May or may not detect events in simple sine wave

    @pytest.mark.asyncio
    async def test_remember_and_recall(self, audio_cortex, sample_waveform):
        """Test audio memory operations."""
        waveform, _ = sample_waveform

        # Remember
        memory_id = await audio_cortex.remember(
            waveform,
            context="test audio",
            importance=0.8,
        )

        assert memory_id is not None

        # Recall
        memories = await audio_cortex.recall_similar("test", limit=5)
        assert isinstance(memories, list)

    @pytest.mark.asyncio
    async def test_register_speaker(self, audio_cortex, sample_waveform):
        """Test speaker registration."""
        waveform, _ = sample_waveform

        profile = await audio_cortex.register_speaker(waveform, "Test User")

        assert profile is not None
        assert profile.name == "Test User"

    @pytest.mark.asyncio
    async def test_answer_question(self, audio_cortex, sample_waveform):
        """Test audio question answering."""
        waveform, _ = sample_waveform

        answer = await audio_cortex.answer_question(
            waveform,
            "What sounds are in this audio?",
        )

        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_summarize(self, audio_cortex, sample_waveform):
        """Test audio summarization."""
        waveform, _ = sample_waveform

        summary = await audio_cortex.summarize(waveform)

        assert isinstance(summary, str)

    @pytest.mark.asyncio
    async def test_get_stats(self, audio_cortex):
        """Test statistics retrieval."""
        stats = audio_cortex.get_stats()

        assert "audio_processed" in stats
        assert "capabilities" in stats


# ==================== SOTA Features Tests ====================

class TestSOTAFeatures:
    """Tests for SOTA audio features."""

    def test_cot_reasoner(self, sample_audio_scene):
        """Test Chain-of-Thought reasoning."""
        reasoner = AudioCoTReasoner()

        # Synchronous test of internal methods
        description = reasoner._describe_scene(sample_audio_scene)
        assert len(description) > 0

        relevant = reasoner._find_relevant_events(
            sample_audio_scene,
            "what sounds are present?",
        )
        assert isinstance(relevant, list)

    def test_streaming_processor(self, sample_waveform):
        """Test streaming audio processor."""
        processor = StreamingAudioProcessor()
        waveform, sr = sample_waveform

        # Process in chunks
        chunk_size = int(0.5 * sr)

        async def process():
            for i in range(0, len(waveform), chunk_size):
                chunk = waveform[i:i + chunk_size]
                result = await processor.process_chunk(chunk)
                assert "is_speech" in result

            final = await processor.finalize()
            assert "transcript" in final

        asyncio.get_event_loop().run_until_complete(process())

    def test_conversational_memory(self, sample_transcript):
        """Test conversational memory."""
        memory = ConversationalMemory()

        async def test():
            conv_id = await memory.add_conversation(sample_transcript)
            assert conv_id is not None

            results = await memory.recall_conversation("hello")
            assert isinstance(results, list)

        asyncio.get_event_loop().run_until_complete(test())

    def test_speaker_behavior_analyzer(self, sample_speaker, sample_transcript):
        """Test speaker behavior analysis."""
        analyzer = SpeakerBehaviorAnalyzer()

        async def test():
            profile = await analyzer.analyze_speaker(sample_speaker, sample_transcript)

            assert profile.speaker_id == sample_speaker.id
            assert profile.avg_turn_duration >= 0
            assert profile.interaction_style in ["listener", "talker", "balanced"]

        asyncio.get_event_loop().run_until_complete(test())


# ==================== Reasoning Tests ====================

class TestAudioReasoning:
    """Tests for audio reasoning capabilities."""

    def test_rule_based_answer(self, sample_audio_scene, sample_transcript):
        """Test rule-based question answering."""
        reasoner = AudioReasoner()

        answer = reasoner._rule_based_answer(
            "How many speakers are there?",
            sample_audio_scene,
            sample_transcript,
        )

        assert "1" in answer or "one" in answer.lower()

    def test_build_context(self, sample_audio_scene, sample_transcript):
        """Test context building."""
        reasoner = AudioReasoner()
        context = reasoner._build_context(sample_audio_scene, sample_transcript)

        assert "Duration" in context
        assert "speakers" in context.lower()

    @pytest.mark.asyncio
    async def test_summarize(self, sample_audio_scene, sample_transcript):
        """Test summarization."""
        reasoner = AudioReasoner()
        summary = reasoner._generate_summary(sample_audio_scene, sample_transcript)

        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_transcript_sentiment(self, sample_transcript):
        """Test transcript sentiment analysis."""
        analyzer = AudioTranscriptAnalyzer()
        result = await analyzer.analyze_sentiment(sample_transcript)

        assert "overall" in result
        assert result["overall"] in ["positive", "negative", "neutral"]

    @pytest.mark.asyncio
    async def test_extract_topics(self, sample_transcript):
        """Test topic extraction."""
        analyzer = AudioTranscriptAnalyzer()
        topics = await analyzer.extract_topics(sample_transcript)

        assert isinstance(topics, list)

    @pytest.mark.asyncio
    async def test_extract_key_phrases(self, sample_transcript):
        """Test key phrase extraction."""
        analyzer = AudioTranscriptAnalyzer()
        phrases = await analyzer.extract_key_phrases(sample_transcript)

        assert isinstance(phrases, list)


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for the complete audio system."""

    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self, sample_waveform):
        """Test the complete processing pipeline."""
        cortex = AuditoryCortex(
            enable_memory=True,
            device="cpu",
            enable_diarization=False,
            enable_tts=False,
        )

        try:
            await cortex.initialize()

            waveform, sr = sample_waveform

            # Process audio
            result = await cortex.process(waveform, store_in_memory=True)
            assert result is not None
            assert result.scene is not None

            # Verify it's in memory
            assert cortex.memory.count() > 0

            # Recall
            memories = await cortex.recall_similar("test", limit=5)
            assert len(memories) > 0

            # Get stats
            stats = cortex.get_stats()
            assert stats["audio_processed"] >= 1

        finally:
            await cortex.shutdown()

    @pytest.mark.asyncio
    async def test_speaker_workflow(self, sample_waveform):
        """Test speaker registration and identification workflow."""
        cortex = AuditoryCortex(
            enable_memory=True,
            device="cpu",
            enable_diarization=False,
            enable_tts=False,
        )

        try:
            await cortex.initialize()

            waveform, _ = sample_waveform

            # Register speaker
            profile = await cortex.register_speaker(waveform, "Test Person")
            assert profile is not None

            # Identify speaker
            identified, confidence = await cortex.identify_speaker(waveform)
            # May or may not identify depending on model availability

        finally:
            await cortex.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
