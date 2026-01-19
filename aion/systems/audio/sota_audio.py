"""
AION SOTA Audio Capabilities

State-of-the-art audio understanding techniques:
- Audio Chain-of-Thought reasoning
- Multi-modal audio-visual fusion
- Conversational memory with topic tracking
- Emotion and sentiment analysis
- Audio source separation
- Real-time streaming analysis
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioEvent,
    AudioEventType,
    AudioScene,
    AudioSegment,
    EmotionalTone,
    Speaker,
    TimeRange,
    Transcript,
    TranscriptSegment,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Audio Chain-of-Thought Reasoning
# ============================================================================

@dataclass
class AudioThought:
    """A single thought in audio reasoning chain."""
    step: int
    observation: str  # What was perceived in the audio
    reasoning: str  # Interpretation and analysis
    evidence: list[str] = field(default_factory=list)  # Supporting evidence
    confidence: float = 0.0
    time_range: Optional[TimeRange] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "observation": self.observation,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "time_range": self.time_range.to_dict() if self.time_range else None,
        }


@dataclass
class AudioCoTResult:
    """Result of Chain-of-Thought audio reasoning."""
    question: str
    answer: str
    thoughts: list[AudioThought]
    final_confidence: float
    audio_id: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "final_confidence": self.final_confidence,
            "audio_id": self.audio_id,
            "processing_time_ms": self.processing_time_ms,
        }


class AudioCoTReasoner:
    """
    Chain-of-Thought reasoning for audio understanding.

    Implements a structured reasoning process:
    1. Segment audio into meaningful chunks
    2. Analyze each segment for relevant information
    3. Build temporal understanding
    4. Synthesize final answer with explicit reasoning trace
    """

    def __init__(self, llm_adapter: Optional[Any] = None):
        """
        Initialize the CoT reasoner.

        Args:
            llm_adapter: Optional LLM adapter for advanced reasoning
        """
        self._llm = llm_adapter

    async def reason(
        self,
        scene: AudioScene,
        question: str,
    ) -> AudioCoTResult:
        """
        Perform chain-of-thought reasoning on audio scene.

        Args:
            scene: Analyzed audio scene
            question: Question to answer

        Returns:
            AudioCoTResult with reasoning trace
        """
        import time
        start_time = time.time()

        thoughts = []
        evidence_collected = []

        # Step 1: Analyze overall scene context
        thought1 = AudioThought(
            step=1,
            observation=f"Audio is {scene.duration:.1f}s long, scene type: {scene.scene_type}",
            reasoning=f"This is a {scene.ambient_description} environment with "
                      f"{scene.event_count} events and {scene.speaker_count} speakers.",
            evidence=[scene.ambient_description, f"{scene.event_count} events"],
            confidence=0.8,
        )
        thoughts.append(thought1)

        # Step 2: Analyze speech content if available
        if scene.transcript and scene.transcript.text:
            thought2 = AudioThought(
                step=2,
                observation=f"Speech detected: '{scene.transcript.text[:200]}...'",
                reasoning=self._analyze_speech_relevance(scene.transcript, question),
                evidence=[f"{scene.transcript.word_count} words",
                         f"language: {scene.transcript.language}"],
                confidence=0.85,
            )
            thoughts.append(thought2)
            evidence_collected.append(scene.transcript.text)

        # Step 3: Analyze events
        if scene.events:
            event_summary = self._summarize_events(scene.events)
            thought3 = AudioThought(
                step=3,
                observation=f"Detected events: {event_summary}",
                reasoning=self._analyze_events_relevance(scene.events, question),
                evidence=[e.label for e in scene.events[:5]],
                confidence=0.75,
            )
            thoughts.append(thought3)
            evidence_collected.extend([e.label for e in scene.events])

        # Step 4: Analyze speakers
        if scene.speakers:
            thought4 = AudioThought(
                step=4,
                observation=f"{len(scene.speakers)} speakers identified",
                reasoning=self._analyze_speakers(scene.speakers, scene.transcript),
                evidence=[s.name or f"Speaker {i+1}" for i, s in enumerate(scene.speakers)],
                confidence=0.8,
            )
            thoughts.append(thought4)

        # Step 5: Music analysis if present
        if scene.has_music and scene.music_analysis:
            ma = scene.music_analysis
            thought5 = AudioThought(
                step=5,
                observation=f"Music detected: {ma.tempo:.0f} BPM, {ma.key or 'unknown key'}",
                reasoning=f"The music has a {ma.mood or 'moderate'} mood with "
                         f"{ma.energy:.1%} energy level.",
                evidence=[f"{ma.tempo:.0f} BPM", ma.key or "unknown", ma.mood or "moderate"],
                confidence=0.7,
            )
            thoughts.append(thought5)

        # Step 6: Synthesize answer
        answer = self._synthesize_answer(thoughts, question, evidence_collected)
        final_confidence = sum(t.confidence for t in thoughts) / max(len(thoughts), 1)

        return AudioCoTResult(
            question=question,
            answer=answer,
            thoughts=thoughts,
            final_confidence=final_confidence,
            audio_id=scene.audio_id,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def _analyze_speech_relevance(self, transcript: Transcript, question: str) -> str:
        """Analyze how speech relates to the question."""
        question_words = set(question.lower().split())
        transcript_words = set(transcript.text.lower().split())
        overlap = question_words & transcript_words

        if overlap:
            return f"Speech contains relevant keywords: {', '.join(list(overlap)[:5])}"
        return "Speech content may provide context for the question."

    def _summarize_events(self, events: list[AudioEvent]) -> str:
        """Create a summary of events."""
        by_type: dict[str, int] = {}
        for event in events:
            label = event.label
            by_type[label] = by_type.get(label, 0) + 1

        parts = [f"{count}x {label}" for label, count in by_type.items()]
        return ", ".join(parts[:5])

    def _analyze_events_relevance(self, events: list[AudioEvent], question: str) -> str:
        """Analyze how events relate to the question."""
        question_lower = question.lower()

        if "sound" in question_lower or "noise" in question_lower:
            return "Events directly relate to question about sounds."
        if "music" in question_lower:
            music_events = [e for e in events if e.event_type == AudioEventType.MUSIC]
            if music_events:
                return f"Found {len(music_events)} music-related events."
        return "Events provide context about the audio environment."

    def _analyze_speakers(
        self,
        speakers: list[Speaker],
        transcript: Optional[Transcript],
    ) -> str:
        """Analyze speaker information."""
        if not speakers:
            return "No speakers identified."

        total_time = sum(s.total_speaking_time for s in speakers)
        analysis = f"Total speaking time: {total_time:.1f}s across {len(speakers)} speakers."

        if transcript and len(speakers) > 1:
            # Check for conversation dynamics
            turn_count = len(transcript.segments)
            analysis += f" {turn_count} speaking turns detected."

        return analysis

    def _synthesize_answer(
        self,
        thoughts: list[AudioThought],
        question: str,
        evidence: list[str],
    ) -> str:
        """Synthesize final answer from reasoning chain."""
        question_lower = question.lower()

        # Build answer based on question type
        if "who" in question_lower:
            speaker_thoughts = [t for t in thoughts if "speaker" in t.observation.lower()]
            if speaker_thoughts:
                return speaker_thoughts[-1].reasoning
            return "Unable to identify specific individuals from the audio."

        if "what" in question_lower:
            speech_thoughts = [t for t in thoughts if "speech" in t.observation.lower()]
            if speech_thoughts:
                return speech_thoughts[-1].reasoning
            event_thoughts = [t for t in thoughts if "event" in t.observation.lower()]
            if event_thoughts:
                return event_thoughts[-1].reasoning
            return "The audio contains: " + ", ".join(evidence[:5]) if evidence else "No specific content identified."

        if "where" in question_lower:
            scene_thoughts = [t for t in thoughts if "scene type" in t.observation.lower()]
            if scene_thoughts:
                return scene_thoughts[0].reasoning
            return "Location cannot be determined from audio alone."

        if "when" in question_lower:
            return "Temporal context would need additional information beyond the audio."

        if "how many" in question_lower:
            for thought in thoughts:
                if "speaker" in thought.observation.lower():
                    return thought.reasoning
                if "event" in thought.observation.lower():
                    return thought.reasoning
            return "Unable to determine quantity from audio."

        # Default: combine evidence
        combined = " ".join(t.reasoning for t in thoughts[:3])
        return combined if combined else "Unable to answer based on available audio information."


# ============================================================================
# Multi-Modal Audio-Visual Fusion
# ============================================================================

@dataclass
class AudioVisualCorrelation:
    """A correlation between audio and visual elements."""
    audio_event: AudioEvent
    visual_element: dict[str, Any]  # From visual cortex
    correlation_type: str  # "sync", "causation", "context"
    confidence: float
    time_offset: float = 0.0  # Audio relative to visual

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_event": self.audio_event.to_dict(),
            "visual_element": self.visual_element,
            "correlation_type": self.correlation_type,
            "confidence": self.confidence,
            "time_offset": self.time_offset,
        }


@dataclass
class AudioVisualScene:
    """Fused audio-visual scene understanding."""
    audio_scene: AudioScene
    visual_scene: Optional[dict[str, Any]] = None  # From visual cortex
    correlations: list[AudioVisualCorrelation] = field(default_factory=list)
    unified_description: str = ""
    dominant_modality: str = "balanced"  # "audio", "visual", "balanced"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_scene": self.audio_scene.to_dict(),
            "visual_scene": self.visual_scene,
            "correlations": [c.to_dict() for c in self.correlations],
            "unified_description": self.unified_description,
            "dominant_modality": self.dominant_modality,
            "timestamp": self.timestamp.isoformat(),
        }


class AudioVisualFusion:
    """
    Fuse audio and visual understanding.

    For video understanding, correlates:
    - Speech with lip movements
    - Sound events with visual actions
    - Music with scene mood
    - Environmental sounds with visual context
    """

    def __init__(self):
        self._correlation_cache: dict[str, list[AudioVisualCorrelation]] = {}

    async def fuse(
        self,
        audio_scene: AudioScene,
        visual_scene: dict[str, Any],
    ) -> AudioVisualScene:
        """
        Fuse audio and visual scene understanding.

        Args:
            audio_scene: Audio scene analysis
            visual_scene: Visual scene analysis (from VisualCortex)

        Returns:
            AudioVisualScene with correlations
        """
        correlations = []

        # Find audio-visual correlations
        if "objects" in visual_scene:
            correlations.extend(
                self._correlate_events_to_objects(
                    audio_scene.events,
                    visual_scene["objects"],
                )
            )

        # Correlate speakers with people
        if audio_scene.speakers and "objects" in visual_scene:
            people = [o for o in visual_scene["objects"]
                     if o.get("label", "").lower() in ["person", "face", "man", "woman"]]
            if people:
                correlations.extend(
                    self._correlate_speakers_to_people(audio_scene.speakers, people)
                )

        # Determine dominant modality
        dominant = self._determine_dominant_modality(audio_scene, visual_scene)

        # Generate unified description
        description = self._generate_unified_description(
            audio_scene, visual_scene, correlations
        )

        return AudioVisualScene(
            audio_scene=audio_scene,
            visual_scene=visual_scene,
            correlations=correlations,
            unified_description=description,
            dominant_modality=dominant,
        )

    def _correlate_events_to_objects(
        self,
        events: list[AudioEvent],
        objects: list[dict[str, Any]],
    ) -> list[AudioVisualCorrelation]:
        """Find correlations between audio events and visual objects."""
        correlations = []

        # Map audio events to visual objects
        audio_visual_mappings = {
            "dog_bark": ["dog"],
            "cat_meow": ["cat"],
            "car": ["car", "vehicle"],
            "engine": ["car", "vehicle", "motorcycle"],
            "speech": ["person", "face"],
            "music": ["instrument", "guitar", "piano"],
            "typing": ["keyboard", "computer", "laptop"],
            "phone": ["phone", "cell phone"],
            "door": ["door"],
        }

        for event in events:
            event_lower = event.label.lower()

            for audio_keyword, visual_keywords in audio_visual_mappings.items():
                if audio_keyword in event_lower:
                    for obj in objects:
                        obj_label = obj.get("label", "").lower()
                        if any(vk in obj_label for vk in visual_keywords):
                            correlations.append(AudioVisualCorrelation(
                                audio_event=event,
                                visual_element=obj,
                                correlation_type="sync",
                                confidence=min(event.confidence, obj.get("confidence", 0.5)),
                            ))

        return correlations

    def _correlate_speakers_to_people(
        self,
        speakers: list[Speaker],
        people: list[dict[str, Any]],
    ) -> list[AudioVisualCorrelation]:
        """Correlate audio speakers to visible people."""
        correlations = []

        # Simple heuristic: match speakers to people by order
        # In a real system, would use lip sync or spatial audio
        for i, (speaker, person) in enumerate(zip(speakers, people)):
            # Create a synthetic event for the correlation
            event = AudioEvent(
                label=f"speech_{speaker.id}",
                event_type=AudioEventType.SPEECH,
                start_time=0.0,
                end_time=speaker.total_speaking_time,
                confidence=speaker.confidence,
            )

            correlations.append(AudioVisualCorrelation(
                audio_event=event,
                visual_element=person,
                correlation_type="context",
                confidence=0.6,  # Lower confidence for heuristic matching
            ))

        return correlations

    def _determine_dominant_modality(
        self,
        audio_scene: AudioScene,
        visual_scene: dict[str, Any],
    ) -> str:
        """Determine which modality provides more information."""
        audio_score = 0.0
        visual_score = 0.0

        # Audio scoring
        if audio_scene.transcript:
            audio_score += 0.4
        audio_score += len(audio_scene.events) * 0.05
        audio_score += len(audio_scene.speakers) * 0.1
        if audio_scene.has_music:
            audio_score += 0.2

        # Visual scoring
        if "objects" in visual_scene:
            visual_score += len(visual_scene["objects"]) * 0.05
        if "caption" in visual_scene:
            visual_score += 0.3
        if "relations" in visual_scene:
            visual_score += len(visual_scene.get("relations", [])) * 0.02

        # Normalize
        total = audio_score + visual_score
        if total == 0:
            return "balanced"

        audio_ratio = audio_score / total
        if audio_ratio > 0.6:
            return "audio"
        elif audio_ratio < 0.4:
            return "visual"
        return "balanced"

    def _generate_unified_description(
        self,
        audio_scene: AudioScene,
        visual_scene: dict[str, Any],
        correlations: list[AudioVisualCorrelation],
    ) -> str:
        """Generate a unified description of the audio-visual scene."""
        parts = []

        # Start with visual context
        if "caption" in visual_scene:
            parts.append(f"Visual: {visual_scene['caption']}")

        # Add audio context
        parts.append(f"Audio: {audio_scene.describe()}")

        # Add correlations
        if correlations:
            corr_desc = []
            for corr in correlations[:3]:
                corr_desc.append(
                    f"{corr.audio_event.label} corresponds to "
                    f"{corr.visual_element.get('label', 'visual element')}"
                )
            if corr_desc:
                parts.append("Correlations: " + "; ".join(corr_desc))

        return " ".join(parts)


# ============================================================================
# Conversational Memory
# ============================================================================

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    emotion: Optional[EmotionalTone] = None
    topics: list[str] = field(default_factory=list)
    intent: Optional[str] = None  # "question", "statement", "command", etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "emotion": self.emotion.value if self.emotion else None,
            "topics": self.topics,
            "intent": self.intent,
        }


@dataclass
class ConversationSummary:
    """Summary of a conversation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    participants: list[str] = field(default_factory=list)
    duration: float = 0.0
    turn_count: int = 0
    main_topics: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "participants": self.participants,
            "duration": self.duration,
            "turn_count": self.turn_count,
            "main_topics": self.main_topics,
            "key_points": self.key_points,
            "action_items": self.action_items,
            "sentiment": self.sentiment,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationTracker:
    """
    Specialized memory for conversations.

    Tracks:
    - Who said what
    - Conversation topics
    - Speaker relationships
    - Temporal flow
    - Questions and answers
    - Action items
    """

    def __init__(self, llm_adapter: Optional[Any] = None):
        """
        Initialize conversation tracker.

        Args:
            llm_adapter: Optional LLM for advanced analysis
        """
        self._llm = llm_adapter
        self._conversations: dict[str, list[ConversationTurn]] = {}
        self._active_conversation: Optional[str] = None
        self._speaker_registry: dict[str, str] = {}  # speaker_id -> name
        self._topic_history: list[str] = []

    async def start_conversation(
        self,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Start tracking a new conversation."""
        conv_id = conversation_id or str(uuid.uuid4())
        self._conversations[conv_id] = []
        self._active_conversation = conv_id
        logger.info("Started conversation tracking", conversation_id=conv_id)
        return conv_id

    async def add_turn(
        self,
        transcript_segment: TranscriptSegment,
        conversation_id: Optional[str] = None,
    ) -> ConversationTurn:
        """
        Add a turn to the conversation.

        Args:
            transcript_segment: Transcribed speech segment
            conversation_id: Conversation to add to

        Returns:
            Created ConversationTurn
        """
        conv_id = conversation_id or self._active_conversation
        if not conv_id or conv_id not in self._conversations:
            conv_id = await self.start_conversation(conv_id)

        # Analyze the turn
        intent = self._detect_intent(transcript_segment.text)
        topics = self._extract_topics(transcript_segment.text)
        emotion = transcript_segment.emotional_tone

        # Get speaker name
        speaker_name = None
        if transcript_segment.speaker_id:
            speaker_name = self._speaker_registry.get(
                transcript_segment.speaker_id,
                f"Speaker {len(self._speaker_registry) + 1}",
            )
            if transcript_segment.speaker_id not in self._speaker_registry:
                self._speaker_registry[transcript_segment.speaker_id] = speaker_name

        turn = ConversationTurn(
            speaker_id=transcript_segment.speaker_id,
            speaker_name=speaker_name,
            text=transcript_segment.text,
            start_time=transcript_segment.start_time,
            end_time=transcript_segment.end_time,
            emotion=emotion,
            topics=topics,
            intent=intent,
        )

        self._conversations[conv_id].append(turn)
        self._topic_history.extend(topics)

        return turn

    async def add_from_transcript(
        self,
        transcript: Transcript,
        conversation_id: Optional[str] = None,
    ) -> list[ConversationTurn]:
        """Add all segments from a transcript."""
        turns = []
        for segment in transcript.segments:
            turn = await self.add_turn(segment, conversation_id)
            turns.append(turn)
        return turns

    def _detect_intent(self, text: str) -> str:
        """Detect the intent of a turn."""
        text_lower = text.lower().strip()

        if text_lower.endswith("?"):
            return "question"
        if any(text_lower.startswith(w) for w in ["please", "can you", "could you", "would you"]):
            return "request"
        if any(w in text_lower for w in ["i think", "in my opinion", "i believe"]):
            return "opinion"
        if any(text_lower.startswith(w) for w in ["okay", "sure", "yes", "no", "agreed"]):
            return "response"
        return "statement"

    def _extract_topics(self, text: str) -> list[str]:
        """Extract topics from text (simple keyword extraction)."""
        # In production, would use NER or topic modeling
        topics = []

        # Common topic indicators
        topic_keywords = {
            "project": ["project", "task", "work", "assignment"],
            "meeting": ["meeting", "call", "discussion", "sync"],
            "deadline": ["deadline", "due", "timeline", "schedule"],
            "budget": ["budget", "cost", "money", "expense", "price"],
            "technical": ["code", "bug", "feature", "system", "api", "database"],
            "personal": ["weekend", "vacation", "family", "health"],
        }

        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        return topics[:3]  # Limit to top 3 topics

    async def end_conversation(
        self,
        conversation_id: Optional[str] = None,
    ) -> ConversationSummary:
        """
        End and summarize a conversation.

        Args:
            conversation_id: Conversation to end

        Returns:
            ConversationSummary
        """
        conv_id = conversation_id or self._active_conversation
        if not conv_id or conv_id not in self._conversations:
            return ConversationSummary()

        turns = self._conversations[conv_id]

        if not turns:
            return ConversationSummary(id=conv_id)

        # Gather statistics
        participants = list(set(
            t.speaker_name or t.speaker_id or "Unknown"
            for t in turns if t.speaker_name or t.speaker_id
        ))

        duration = turns[-1].end_time - turns[0].start_time

        # Gather all topics
        all_topics = []
        for turn in turns:
            all_topics.extend(turn.topics)
        topic_counts: dict[str, int] = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        main_topics = sorted(topic_counts.keys(), key=lambda t: topic_counts[t], reverse=True)[:5]

        # Extract key points (turns with questions/requests)
        key_points = []
        for turn in turns:
            if turn.intent in ["question", "request"]:
                key_points.append(f"{turn.speaker_name or 'Speaker'}: {turn.text[:100]}")

        # Look for action items
        action_items = []
        action_phrases = ["will do", "i'll", "we'll", "going to", "need to", "should", "must"]
        for turn in turns:
            if any(phrase in turn.text.lower() for phrase in action_phrases):
                action_items.append(f"{turn.speaker_name or 'Speaker'}: {turn.text[:100]}")

        # Determine sentiment
        emotions = [t.emotion for t in turns if t.emotion]
        if emotions:
            positive = sum(1 for e in emotions if e in [EmotionalTone.HAPPY, EmotionalTone.EXCITED])
            negative = sum(1 for e in emotions if e in [EmotionalTone.SAD, EmotionalTone.ANGRY, EmotionalTone.FEARFUL])
            if positive > negative * 2:
                sentiment = "positive"
            elif negative > positive * 2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        else:
            sentiment = "neutral"

        # Generate title
        title = f"Conversation about {', '.join(main_topics[:2])}" if main_topics else "Conversation"

        summary = ConversationSummary(
            id=conv_id,
            title=title,
            participants=participants,
            duration=duration,
            turn_count=len(turns),
            main_topics=main_topics,
            key_points=key_points[:5],
            action_items=action_items[:5],
            sentiment=sentiment,
        )

        if conv_id == self._active_conversation:
            self._active_conversation = None

        return summary

    async def search_conversation(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
    ) -> list[ConversationTurn]:
        """
        Search for turns matching a query.

        Args:
            query: Text query
            conversation_id: Specific conversation (None = all)
            speaker_id: Filter by speaker

        Returns:
            List of matching ConversationTurn
        """
        query_words = set(query.lower().split())
        results = []

        conversations = (
            {conversation_id: self._conversations[conversation_id]}
            if conversation_id and conversation_id in self._conversations
            else self._conversations
        )

        for conv_id, turns in conversations.items():
            for turn in turns:
                if speaker_id and turn.speaker_id != speaker_id:
                    continue

                turn_words = set(turn.text.lower().split())
                if query_words & turn_words:
                    results.append(turn)

        return results

    async def get_speaker_turns(
        self,
        speaker_id: str,
        conversation_id: Optional[str] = None,
    ) -> list[ConversationTurn]:
        """Get all turns by a specific speaker."""
        results = []

        conversations = (
            {conversation_id: self._conversations[conversation_id]}
            if conversation_id and conversation_id in self._conversations
            else self._conversations
        )

        for conv_id, turns in conversations.items():
            for turn in turns:
                if turn.speaker_id == speaker_id:
                    results.append(turn)

        return results

    def get_recent_topics(self, limit: int = 10) -> list[str]:
        """Get recent topics discussed."""
        return list(reversed(self._topic_history[-limit:]))

    def register_speaker(self, speaker_id: str, name: str) -> None:
        """Register a speaker name."""
        self._speaker_registry[speaker_id] = name

    def get_stats(self) -> dict[str, Any]:
        """Get conversation tracker statistics."""
        total_turns = sum(len(turns) for turns in self._conversations.values())
        return {
            "total_conversations": len(self._conversations),
            "active_conversation": self._active_conversation,
            "total_turns": total_turns,
            "registered_speakers": len(self._speaker_registry),
            "topic_history_size": len(self._topic_history),
        }


# ============================================================================
# Emotion Analysis
# ============================================================================

@dataclass
class EmotionAnalysisResult:
    """Result of emotion analysis."""
    primary_emotion: EmotionalTone
    emotion_scores: dict[str, float]  # emotion -> confidence
    arousal: float  # Low (0) to high (1) activation
    valence: float  # Negative (0) to positive (1)
    dominant_emotion_confidence: float
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion.value,
            "emotion_scores": self.emotion_scores,
            "arousal": self.arousal,
            "valence": self.valence,
            "dominant_emotion_confidence": self.dominant_emotion_confidence,
            "timeline": self.timeline,
        }


class EmotionAnalyzer:
    """
    Analyze emotion and sentiment in audio.

    Uses acoustic features and optional text sentiment
    to determine emotional content.
    """

    def __init__(self):
        self._emotion_model = None  # Would load a trained model

    async def analyze(
        self,
        audio_scene: AudioScene,
    ) -> EmotionAnalysisResult:
        """
        Analyze emotion in audio scene.

        Args:
            audio_scene: Analyzed audio scene

        Returns:
            EmotionAnalysisResult
        """
        # Initialize scores
        emotion_scores = {e.value: 0.0 for e in EmotionalTone}

        # Use scene's emotional tone if available
        if audio_scene.emotional_tone:
            emotion_scores[audio_scene.emotional_tone.value] = audio_scene.emotional_confidence or 0.7

        # Analyze based on acoustic features
        arousal = 0.5
        valence = 0.5

        # Music influence
        if audio_scene.has_music and audio_scene.music_analysis:
            ma = audio_scene.music_analysis
            valence = ma.valence
            arousal = ma.energy

            # Map music mood to emotions
            if ma.mood == "energetic":
                emotion_scores["excited"] += 0.3
                emotion_scores["happy"] += 0.2
            elif ma.mood == "calm":
                emotion_scores["calm"] += 0.4
            elif ma.mood == "melancholic":
                emotion_scores["sad"] += 0.3

        # Speech influence
        if audio_scene.transcript:
            text_emotions = self._analyze_text_sentiment(audio_scene.transcript.text)
            for emotion, score in text_emotions.items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += score * 0.5

        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}

        # Find primary emotion
        primary = max(emotion_scores.items(), key=lambda x: x[1])
        primary_emotion = EmotionalTone(primary[0])
        confidence = primary[1]

        # Build timeline (simplified - would analyze segments)
        timeline = []
        if audio_scene.transcript and audio_scene.transcript.segments:
            for seg in audio_scene.transcript.segments[:5]:
                seg_emotions = self._analyze_text_sentiment(seg.text)
                primary_seg = max(seg_emotions.items(), key=lambda x: x[1]) if seg_emotions else ("neutral", 0.5)
                timeline.append({
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "emotion": primary_seg[0],
                    "confidence": primary_seg[1],
                })

        return EmotionAnalysisResult(
            primary_emotion=primary_emotion,
            emotion_scores=emotion_scores,
            arousal=arousal,
            valence=valence,
            dominant_emotion_confidence=confidence,
            timeline=timeline,
        )

    def _analyze_text_sentiment(self, text: str) -> dict[str, float]:
        """Simple text sentiment analysis."""
        text_lower = text.lower()
        emotions: dict[str, float] = {}

        # Simple keyword-based emotion detection
        emotion_keywords = {
            "happy": ["happy", "great", "wonderful", "excellent", "love", "excited", "amazing"],
            "sad": ["sad", "sorry", "unfortunately", "miss", "lost", "disappointed"],
            "angry": ["angry", "frustrated", "annoyed", "furious", "hate"],
            "fearful": ["scared", "afraid", "worried", "nervous", "anxious"],
            "surprised": ["surprised", "wow", "amazing", "unexpected", "shocked"],
            "neutral": ["okay", "fine", "alright", "sure", "yes", "no"],
        }

        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                emotions[emotion] = min(count * 0.3, 0.9)

        if not emotions:
            emotions["neutral"] = 0.5

        return emotions


# ============================================================================
# SOTA Audio Cortex (Combined)
# ============================================================================

class SOTAAudioCortex:
    """
    State-of-the-art audio cortex with advanced capabilities.

    Combines:
    - Chain-of-thought reasoning
    - Audio-visual fusion
    - Conversation tracking
    - Emotion analysis
    """

    def __init__(self, llm_adapter: Optional[Any] = None):
        """
        Initialize SOTA audio cortex.

        Args:
            llm_adapter: Optional LLM for advanced reasoning
        """
        self._llm = llm_adapter
        self._cot_reasoner = AudioCoTReasoner(llm_adapter)
        self._av_fusion = AudioVisualFusion()
        self._conversation_tracker = ConversationTracker(llm_adapter)
        self._emotion_analyzer = EmotionAnalyzer()

    async def reason_about_audio(
        self,
        audio_scene: AudioScene,
        question: str,
    ) -> AudioCoTResult:
        """
        Perform chain-of-thought reasoning about audio.

        Args:
            audio_scene: Analyzed audio scene
            question: Question to answer

        Returns:
            AudioCoTResult with reasoning trace
        """
        return await self._cot_reasoner.reason(audio_scene, question)

    async def fuse_with_visual(
        self,
        audio_scene: AudioScene,
        visual_scene: dict[str, Any],
    ) -> AudioVisualScene:
        """
        Fuse audio with visual scene understanding.

        Args:
            audio_scene: Audio scene analysis
            visual_scene: Visual scene from VisualCortex

        Returns:
            Fused AudioVisualScene
        """
        return await self._av_fusion.fuse(audio_scene, visual_scene)

    async def track_conversation(
        self,
        transcript: Transcript,
    ) -> ConversationSummary:
        """
        Track and analyze a conversation.

        Args:
            transcript: Transcript with speaker diarization

        Returns:
            ConversationSummary
        """
        conv_id = await self._conversation_tracker.start_conversation()
        await self._conversation_tracker.add_from_transcript(transcript, conv_id)
        return await self._conversation_tracker.end_conversation(conv_id)

    async def analyze_emotion(
        self,
        audio_scene: AudioScene,
    ) -> EmotionAnalysisResult:
        """
        Analyze emotion in audio.

        Args:
            audio_scene: Analyzed audio scene

        Returns:
            EmotionAnalysisResult
        """
        return await self._emotion_analyzer.analyze(audio_scene)

    @property
    def conversation_tracker(self) -> ConversationTracker:
        """Access the conversation tracker."""
        return self._conversation_tracker

    def get_stats(self) -> dict[str, Any]:
        """Get SOTA cortex statistics."""
        return {
            "conversation_tracker": self._conversation_tracker.get_stats(),
        }
