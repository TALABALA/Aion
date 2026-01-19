"""
AION SOTA Audio System

State-of-the-art audio processing with:
- Audio Chain-of-Thought reasoning
- Multi-modal audio-visual fusion
- Conversational audio memory
- Temporal event reasoning
- Speaker behavior analysis
- Real-time streaming support
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union, Callable
from pathlib import Path

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioScene,
    AudioEvent,
    Transcript,
    Speaker,
    TimeRange,
    AudioSegment,
    VoiceProfile,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Audio Chain-of-Thought Reasoning
# ============================================================================

@dataclass
class AudioThought:
    """A thought in audio reasoning."""
    step: int
    observation: str
    reasoning: str
    conclusion: Optional[str] = None
    time_references: list[TimeRange] = field(default_factory=list)
    confidence: float = 0.0


class AudioCoTReasoner:
    """
    Chain-of-Thought reasoning for audio understanding.

    Steps:
    1. Segment audio into meaningful chunks
    2. Analyze each segment
    3. Build temporal understanding
    4. Synthesize final answer
    """

    def __init__(self, llm_adapter=None):
        self.llm = llm_adapter
        self.max_steps = 5

    async def reason(
        self,
        scene: AudioScene,
        question: str,
    ) -> tuple[str, list[AudioThought]]:
        """
        Perform chain-of-thought reasoning about audio.

        Args:
            scene: Audio scene analysis
            question: Question to answer

        Returns:
            Tuple of (answer, reasoning_trace)
        """
        thoughts = []

        # Step 1: Identify relevant elements
        thought1 = AudioThought(
            step=1,
            observation=self._describe_scene(scene),
            reasoning=f"Analyzing the audio scene to answer: {question}",
            confidence=0.9,
        )
        thoughts.append(thought1)

        # Step 2: Identify key events
        relevant_events = self._find_relevant_events(scene, question)
        if relevant_events:
            thought2 = AudioThought(
                step=2,
                observation=f"Found {len(relevant_events)} relevant events: "
                           f"{', '.join(e.label for e in relevant_events[:5])}",
                reasoning="These events may help answer the question",
                time_references=[e.time_range for e in relevant_events[:3]],
                confidence=0.8,
            )
            thoughts.append(thought2)

        # Step 3: Analyze speech if present
        if scene.transcript:
            thought3 = AudioThought(
                step=3,
                observation=f"Transcript: {scene.transcript.text[:200]}...",
                reasoning="Analyzing what was said",
                confidence=0.85,
            )
            thoughts.append(thought3)

        # Step 4: Consider context
        context_thought = AudioThought(
            step=4,
            observation=f"Environment: {scene.ambient_description or 'unknown'}",
            reasoning="Considering the context of the audio",
            confidence=0.7,
        )
        thoughts.append(context_thought)

        # Step 5: Synthesize answer
        answer = self._synthesize_answer(scene, question, thoughts)
        final_thought = AudioThought(
            step=5,
            observation="Synthesizing final answer",
            reasoning="Combining all observations",
            conclusion=answer,
            confidence=0.8,
        )
        thoughts.append(final_thought)

        return answer, thoughts

    def _describe_scene(self, scene: AudioScene) -> str:
        """Generate scene description."""
        parts = []
        parts.append(f"Duration: {scene.duration:.1f}s")
        parts.append(f"Events: {len(scene.events)}")
        parts.append(f"Speakers: {len(scene.speakers)}")
        if scene.music_analysis:
            parts.append("Contains music")
        if scene.emotional_tone:
            parts.append(f"Tone: {scene.emotional_tone}")
        return "; ".join(parts)

    def _find_relevant_events(
        self,
        scene: AudioScene,
        question: str,
    ) -> list[AudioEvent]:
        """Find events relevant to the question."""
        q_lower = question.lower()

        relevant = []
        for event in scene.events:
            # Simple keyword matching
            if event.label.lower() in q_lower:
                relevant.append(event)
            elif event.category in q_lower:
                relevant.append(event)
            elif event.confidence > 0.8:
                relevant.append(event)

        return sorted(relevant, key=lambda e: e.confidence, reverse=True)

    def _synthesize_answer(
        self,
        scene: AudioScene,
        question: str,
        thoughts: list[AudioThought],
    ) -> str:
        """Synthesize final answer from thoughts."""
        q_lower = question.lower()

        # Answer based on question type
        if "how many" in q_lower:
            if "speaker" in q_lower:
                return f"There are {len(scene.speakers)} speakers in the audio."
            if "event" in q_lower or "sound" in q_lower:
                return f"There are {len(scene.events)} distinct audio events."

        if "who" in q_lower and "speak" in q_lower:
            if scene.speakers:
                names = [s.name or f"Speaker {i+1}" for i, s in enumerate(scene.speakers)]
                return f"The following people spoke: {', '.join(names)}"
            return "No speakers were identified."

        if "what" in q_lower:
            if "music" in q_lower and scene.music_analysis:
                m = scene.music_analysis
                return f"The music is {m.mood or 'background'} in {m.key} {m.mode} at {m.tempo_bpm:.0f} BPM."
            if "sound" in q_lower or "hear" in q_lower:
                if scene.events:
                    labels = list(set(e.label for e in scene.events[:5]))
                    return f"The following sounds are present: {', '.join(labels)}"

        if "when" in q_lower:
            # Find relevant events and their times
            relevant = self._find_relevant_events(scene, question)
            if relevant:
                times = [f"{e.label} at {e.start_time:.1f}s" for e in relevant[:3]]
                return f"Events occurred: {', '.join(times)}"

        if "where" in q_lower:
            return f"Based on the audio, this appears to be {scene.ambient_description or 'an indoor environment'}."

        # Default to scene description
        return scene.describe()


# ============================================================================
# Audio-Visual Fusion
# ============================================================================

@dataclass
class AudioVisualAlignment:
    """Alignment between audio and visual elements."""
    audio_event_id: str
    visual_element_id: str
    alignment_type: str  # "speech_lip_sync", "sound_action", "music_scene"
    confidence: float
    time_range: TimeRange


class AudioVisualFusion:
    """
    Fuse audio and visual understanding.

    For video understanding, correlate:
    - Speech with lip movements
    - Sound events with visual actions
    - Music with scene mood
    """

    def __init__(self):
        self._alignments: list[AudioVisualAlignment] = []

    async def fuse(
        self,
        audio_scene: AudioScene,
        visual_scene: Any,  # SceneGraph from vision system
    ) -> dict[str, Any]:
        """
        Fuse audio and visual scene understanding.

        Args:
            audio_scene: Audio scene analysis
            visual_scene: Visual scene graph

        Returns:
            Fused understanding
        """
        alignments = []

        # Align speech with detected people
        if hasattr(visual_scene, 'objects'):
            people = [o for o in visual_scene.objects if o.label == "person"]

            for speaker in audio_scene.speakers:
                # Match speakers to visible people
                if people:
                    # Simple assignment by order
                    person_idx = len(alignments) % len(people)
                    alignments.append(AudioVisualAlignment(
                        audio_event_id=speaker.id,
                        visual_element_id=people[person_idx].id,
                        alignment_type="speech_lip_sync",
                        confidence=0.7,
                        time_range=speaker.segments[0] if speaker.segments else TimeRange(0, 0),
                    ))

            # Align sound events with actions
            for event in audio_scene.events:
                if event.category == "action":
                    # Look for related objects
                    for obj in visual_scene.objects:
                        if self._is_related(event.label, obj.label):
                            alignments.append(AudioVisualAlignment(
                                audio_event_id=event.id,
                                visual_element_id=obj.id,
                                alignment_type="sound_action",
                                confidence=0.6,
                                time_range=event.time_range,
                            ))
                            break

        self._alignments = alignments

        return {
            "alignments": [self._alignment_to_dict(a) for a in alignments],
            "audio_description": audio_scene.describe(),
            "combined_narrative": self._generate_narrative(audio_scene, visual_scene),
        }

    def _is_related(self, audio_label: str, visual_label: str) -> bool:
        """Check if audio and visual labels are related."""
        relations = {
            "dog_bark": ["dog"],
            "cat_meow": ["cat"],
            "car": ["car", "vehicle"],
            "door": ["door"],
            "phone_ringing": ["phone", "cell phone"],
            "keyboard_typing": ["keyboard", "laptop", "computer"],
        }

        related = relations.get(audio_label.lower(), [])
        return visual_label.lower() in related

    def _alignment_to_dict(self, alignment: AudioVisualAlignment) -> dict:
        return {
            "audio_event_id": alignment.audio_event_id,
            "visual_element_id": alignment.visual_element_id,
            "alignment_type": alignment.alignment_type,
            "confidence": alignment.confidence,
            "time_range": alignment.time_range.to_dict(),
        }

    def _generate_narrative(
        self,
        audio_scene: AudioScene,
        visual_scene: Any,
    ) -> str:
        """Generate combined audio-visual narrative."""
        parts = []

        # Visual description if available
        if hasattr(visual_scene, 'global_features'):
            caption = visual_scene.global_features.get('caption', '')
            if caption:
                parts.append(f"Visual: {caption}")

        # Audio description
        parts.append(f"Audio: {audio_scene.describe()}")

        # Combined insights
        if audio_scene.has_speech and hasattr(visual_scene, 'objects'):
            people = [o for o in visual_scene.objects if o.label == "person"]
            if people:
                parts.append(
                    f"There are {len(people)} visible people and "
                    f"{len(audio_scene.speakers)} speakers."
                )

        return " ".join(parts)


# ============================================================================
# Conversational Audio Memory
# ============================================================================

@dataclass
class ConversationTurn:
    """A turn in a conversation."""
    speaker_id: str
    text: str
    time_range: TimeRange
    sentiment: Optional[str] = None
    topics: list[str] = field(default_factory=list)


@dataclass
class ConversationSummary:
    """Summary of a conversation."""
    id: str
    participants: list[str]
    main_topics: list[str]
    duration: float
    turn_count: int
    key_points: list[str]
    sentiment_flow: list[str]
    created_at: datetime = field(default_factory=datetime.now)


class ConversationalMemory:
    """
    Specialized memory for conversations.

    Tracks:
    - Who said what
    - Conversation topics
    - Speaker relationships
    - Temporal flow
    - Sentiment progression
    """

    def __init__(self):
        self._conversations: dict[str, list[ConversationTurn]] = {}
        self._summaries: dict[str, ConversationSummary] = {}
        self._speaker_relations: dict[str, dict[str, float]] = {}

    async def add_conversation(
        self,
        transcript: Transcript,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Add a conversation to memory.

        Args:
            transcript: Conversation transcript
            context: Additional context

        Returns:
            Conversation ID
        """
        conv_id = str(uuid.uuid4())
        turns = []

        for segment in transcript.segments:
            turn = ConversationTurn(
                speaker_id=segment.speaker_id or "unknown",
                text=segment.text,
                time_range=segment.time_range,
                topics=self._extract_topics(segment.text),
            )
            turns.append(turn)

        self._conversations[conv_id] = turns

        # Update speaker relations
        speakers = list(set(t.speaker_id for t in turns))
        for s1 in speakers:
            for s2 in speakers:
                if s1 != s2:
                    self._update_relation(s1, s2, len(turns))

        # Generate summary
        summary = await self._summarize_conversation(conv_id, turns, transcript)
        self._summaries[conv_id] = summary

        return conv_id

    def _extract_topics(self, text: str) -> list[str]:
        """Extract topics from text using simple keyword extraction."""
        # Common topic keywords
        topic_keywords = {
            "work": ["work", "office", "meeting", "project", "deadline"],
            "family": ["family", "kids", "children", "parent", "home"],
            "health": ["health", "doctor", "sick", "medicine", "hospital"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel"],
            "food": ["food", "restaurant", "dinner", "lunch", "eat"],
            "technology": ["computer", "phone", "app", "software", "tech"],
            "money": ["money", "pay", "cost", "price", "budget"],
        }

        text_lower = text.lower()
        found_topics = []

        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_topics.append(topic)

        return found_topics

    def _update_relation(
        self,
        speaker1: str,
        speaker2: str,
        interaction_count: int,
    ) -> None:
        """Update relationship strength between speakers."""
        if speaker1 not in self._speaker_relations:
            self._speaker_relations[speaker1] = {}

        current = self._speaker_relations[speaker1].get(speaker2, 0)
        self._speaker_relations[speaker1][speaker2] = current + interaction_count * 0.1

    async def _summarize_conversation(
        self,
        conv_id: str,
        turns: list[ConversationTurn],
        transcript: Transcript,
    ) -> ConversationSummary:
        """Generate conversation summary."""
        participants = list(set(t.speaker_id for t in turns))

        # Collect all topics
        all_topics = []
        for turn in turns:
            all_topics.extend(turn.topics)

        # Count topic frequency
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        main_topics = sorted(topic_counts.keys(), key=lambda t: topic_counts[t], reverse=True)[:5]

        # Extract key points (simple: first sentence of each speaker's first turn)
        key_points = []
        seen_speakers = set()
        for turn in turns:
            if turn.speaker_id not in seen_speakers:
                first_sentence = turn.text.split('.')[0]
                if first_sentence:
                    key_points.append(f"{turn.speaker_id}: {first_sentence}")
                seen_speakers.add(turn.speaker_id)

        return ConversationSummary(
            id=conv_id,
            participants=participants,
            main_topics=main_topics,
            duration=transcript.duration,
            turn_count=len(turns),
            key_points=key_points[:5],
            sentiment_flow=[],
        )

    async def recall_conversation(
        self,
        query: str,
        speaker: Optional[str] = None,
        limit: int = 5,
    ) -> list[ConversationSummary]:
        """
        Recall conversations matching query.

        Args:
            query: Search query
            speaker: Filter by speaker
            limit: Maximum results

        Returns:
            List of matching conversation summaries
        """
        results = []
        query_lower = query.lower()

        for conv_id, summary in self._summaries.items():
            score = 0.0

            # Match topics
            for topic in summary.main_topics:
                if topic in query_lower:
                    score += 1.0

            # Match speaker
            if speaker and speaker in summary.participants:
                score += 0.5

            # Match key points
            for point in summary.key_points:
                if any(word in point.lower() for word in query_lower.split()):
                    score += 0.3

            if score > 0:
                results.append((summary, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]

    def get_speaker_relations(self, speaker_id: str) -> dict[str, float]:
        """Get relationship strengths for a speaker."""
        return self._speaker_relations.get(speaker_id, {})


# ============================================================================
# Streaming Audio Processing
# ============================================================================

class StreamingAudioProcessor:
    """
    Real-time streaming audio processing.

    Supports:
    - Continuous speech recognition
    - Live event detection
    - Real-time speaker tracking
    - Voice activity detection
    """

    def __init__(
        self,
        chunk_duration: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate

        # Buffer for accumulating audio
        self._buffer: list[np.ndarray] = []
        self._buffer_duration = 0.0

        # State tracking
        self._current_speaker: Optional[str] = None
        self._is_speech_active = False
        self._accumulated_transcript = ""

        # Callbacks
        self._on_transcript: Optional[Callable] = None
        self._on_event: Optional[Callable] = None
        self._on_speaker_change: Optional[Callable] = None

    def set_callbacks(
        self,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[AudioEvent], None]] = None,
        on_speaker_change: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Set callback functions for real-time events."""
        self._on_transcript = on_transcript
        self._on_event = on_event
        self._on_speaker_change = on_speaker_change

    async def process_chunk(
        self,
        audio_chunk: np.ndarray,
    ) -> dict[str, Any]:
        """
        Process a chunk of streaming audio.

        Args:
            audio_chunk: Audio samples

        Returns:
            Processing results for this chunk
        """
        # Add to buffer
        self._buffer.append(audio_chunk)
        self._buffer_duration += len(audio_chunk) / self.sample_rate

        # Voice activity detection
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        is_speech = rms > 0.02  # Simple threshold

        results = {
            "chunk_duration": len(audio_chunk) / self.sample_rate,
            "is_speech": is_speech,
            "rms_level": float(rms),
        }

        # Detect speech state changes
        if is_speech and not self._is_speech_active:
            self._is_speech_active = True
            results["speech_started"] = True

        elif not is_speech and self._is_speech_active:
            # Speech ended - process accumulated buffer
            if len(self._buffer) > 0:
                full_audio = np.concatenate(self._buffer)
                # Would trigger transcription here
                results["speech_ended"] = True
                results["speech_duration"] = self._buffer_duration

                # Clear buffer
                self._buffer = []
                self._buffer_duration = 0.0

            self._is_speech_active = False

        return results

    async def finalize(self) -> dict[str, Any]:
        """
        Finalize processing and return accumulated results.

        Returns:
            Final processing results
        """
        results = {
            "transcript": self._accumulated_transcript,
            "buffer_duration": self._buffer_duration,
        }

        # Clear state
        self._buffer = []
        self._buffer_duration = 0.0
        self._accumulated_transcript = ""

        return results


# ============================================================================
# Speaker Behavior Analysis
# ============================================================================

@dataclass
class SpeakerBehaviorProfile:
    """Behavioral profile of a speaker."""
    speaker_id: str
    avg_turn_duration: float
    speaking_rate_wpm: float  # Words per minute
    interruption_tendency: float  # 0-1 scale
    dominant_topics: list[str]
    sentiment_tendency: str  # "positive", "negative", "neutral"
    interaction_style: str  # "listener", "talker", "balanced"


class SpeakerBehaviorAnalyzer:
    """
    Analyze speaker behavior patterns.

    Tracks:
    - Speaking patterns
    - Interaction style
    - Emotional tendencies
    - Topic preferences
    """

    def __init__(self):
        self._profiles: dict[str, SpeakerBehaviorProfile] = {}
        self._turn_history: dict[str, list[float]] = {}  # speaker -> turn durations
        self._word_counts: dict[str, int] = {}

    async def analyze_speaker(
        self,
        speaker: Speaker,
        transcript: Transcript,
    ) -> SpeakerBehaviorProfile:
        """
        Analyze speaker behavior from transcript.

        Args:
            speaker: Speaker to analyze
            transcript: Transcript with speaker's segments

        Returns:
            SpeakerBehaviorProfile
        """
        speaker_segments = [
            seg for seg in transcript.segments
            if seg.speaker_id == speaker.id
        ]

        if not speaker_segments:
            return SpeakerBehaviorProfile(
                speaker_id=speaker.id,
                avg_turn_duration=0,
                speaking_rate_wpm=0,
                interruption_tendency=0,
                dominant_topics=[],
                sentiment_tendency="neutral",
                interaction_style="balanced",
            )

        # Calculate metrics
        turn_durations = [seg.duration for seg in speaker_segments]
        avg_turn_duration = sum(turn_durations) / len(turn_durations)

        total_words = sum(seg.word_count for seg in speaker_segments)
        total_time = sum(turn_durations)
        speaking_rate = (total_words / total_time * 60) if total_time > 0 else 0

        # Estimate interruption tendency
        interruptions = 0
        for i, seg in enumerate(transcript.segments[1:], 1):
            if seg.speaker_id == speaker.id:
                prev_seg = transcript.segments[i - 1]
                if prev_seg.speaker_id != speaker.id:
                    if seg.start_time < prev_seg.end_time + 0.2:
                        interruptions += 1

        interruption_tendency = interruptions / len(speaker_segments) if speaker_segments else 0

        # Determine interaction style
        total_speaking_time = sum(seg.duration for seg in speaker_segments)
        total_duration = transcript.duration
        speaking_ratio = total_speaking_time / total_duration if total_duration > 0 else 0

        if speaking_ratio > 0.6:
            interaction_style = "talker"
        elif speaking_ratio < 0.3:
            interaction_style = "listener"
        else:
            interaction_style = "balanced"

        profile = SpeakerBehaviorProfile(
            speaker_id=speaker.id,
            avg_turn_duration=avg_turn_duration,
            speaking_rate_wpm=speaking_rate,
            interruption_tendency=min(1.0, interruption_tendency),
            dominant_topics=[],  # Would need topic extraction
            sentiment_tendency="neutral",
            interaction_style=interaction_style,
        )

        self._profiles[speaker.id] = profile
        return profile

    def get_profile(self, speaker_id: str) -> Optional[SpeakerBehaviorProfile]:
        """Get stored behavior profile."""
        return self._profiles.get(speaker_id)


# ============================================================================
# SOTA Audio Cortex
# ============================================================================

class SOTAAudioCortex:
    """
    State-of-the-art audio system combining:
    - Audio Chain-of-Thought reasoning
    - Audio-visual fusion
    - Conversational memory
    - Speaker behavior analysis
    - Streaming support
    """

    def __init__(self, llm_adapter=None, base_audio_system=None):
        self.llm = llm_adapter
        self.base_audio = base_audio_system

        # Advanced components
        self.cot_reasoner = AudioCoTReasoner(llm_adapter)
        self.av_fusion = AudioVisualFusion()
        self.conversation_memory = ConversationalMemory()
        self.behavior_analyzer = SpeakerBehaviorAnalyzer()
        self.streaming_processor = StreamingAudioProcessor()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the SOTA audio cortex."""
        if self.base_audio:
            await self.base_audio.initialize()
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the SOTA audio cortex."""
        if self.base_audio:
            await self.base_audio.shutdown()
        self._initialized = False

    async def analyze_with_reasoning(
        self,
        audio: Union[str, Path, np.ndarray],
        question: str,
    ) -> dict[str, Any]:
        """
        Analyze audio with full reasoning trace.

        Args:
            audio: Audio source
            question: Question to answer

        Returns:
            Analysis results with reasoning
        """
        # Get base analysis
        if self.base_audio:
            result = await self.base_audio.process(audio, query=question)
            scene = result.scene
        else:
            # Mock scene
            scene = AudioScene(
                id="mock",
                audio_id="mock",
                duration=0,
                events=[],
                speakers=[],
            )

        # Perform CoT reasoning
        answer, thoughts = await self.cot_reasoner.reason(scene, question)

        return {
            "answer": answer,
            "reasoning_steps": [
                {
                    "step": t.step,
                    "observation": t.observation,
                    "reasoning": t.reasoning,
                    "conclusion": t.conclusion,
                    "confidence": t.confidence,
                }
                for t in thoughts
            ],
            "scene": scene.to_dict(),
        }

    async def analyze_conversation(
        self,
        audio: Union[str, Path, np.ndarray],
    ) -> dict[str, Any]:
        """
        Deep conversation analysis.

        Args:
            audio: Audio containing conversation

        Returns:
            Conversation analysis
        """
        if not self.base_audio:
            return {"error": "Base audio system not available"}

        # Get transcription with diarization
        result = await self.base_audio.process(audio)
        transcript = result.transcript

        if not transcript:
            return {"error": "Could not transcribe audio"}

        # Add to conversation memory
        conv_id = await self.conversation_memory.add_conversation(transcript)

        # Analyze each speaker's behavior
        speaker_profiles = []
        for speaker in transcript.speakers:
            profile = await self.behavior_analyzer.analyze_speaker(speaker, transcript)
            speaker_profiles.append({
                "speaker_id": profile.speaker_id,
                "avg_turn_duration": profile.avg_turn_duration,
                "speaking_rate_wpm": profile.speaking_rate_wpm,
                "interaction_style": profile.interaction_style,
                "interruption_tendency": profile.interruption_tendency,
            })

        return {
            "conversation_id": conv_id,
            "transcript": transcript.to_dict(),
            "speaker_profiles": speaker_profiles,
            "summary": self.conversation_memory._summaries.get(conv_id).key_points if conv_id in self.conversation_memory._summaries else [],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get SOTA audio cortex statistics."""
        return {
            "initialized": self._initialized,
            "has_base_audio": self.base_audio is not None,
            "conversations_stored": len(self.conversation_memory._conversations),
            "speaker_profiles": len(self.behavior_analyzer._profiles),
        }
