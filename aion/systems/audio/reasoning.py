"""
AION Audio Reasoning

LLM-integrated audio reasoning capabilities:
- Audio question answering
- Audio summarization
- Audio-to-text generation
- Multi-turn audio dialogue
- Audio captioning
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioScene,
    Transcript,
    TranscriptSegment,
    Speaker,
    AudioEvent,
    MusicAnalysis,
    TimeRange,
)

logger = structlog.get_logger(__name__)


@dataclass
class AudioDialogueTurn:
    """A turn in an audio dialogue session."""
    role: str  # "user" or "assistant"
    content: str
    audio_context: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AudioReasoner:
    """
    LLM-based audio reasoning.

    Integrates with LLM adapters for:
    - Complex audio question answering
    - Audio content summarization
    - Audio-to-text generation
    - Multi-turn dialogue about audio
    """

    def __init__(self, llm_adapter=None):
        self.llm = llm_adapter
        self._dialogue_history: list[AudioDialogueTurn] = []
        self._current_scene: Optional[AudioScene] = None
        self._current_transcript: Optional[Transcript] = None

    def set_context(
        self,
        scene: Optional[AudioScene] = None,
        transcript: Optional[Transcript] = None,
    ) -> None:
        """Set the current audio context for reasoning."""
        self._current_scene = scene
        self._current_transcript = transcript

    def clear_context(self) -> None:
        """Clear audio context and dialogue history."""
        self._current_scene = None
        self._current_transcript = None
        self._dialogue_history = []

    async def answer_question(
        self,
        question: str,
        scene: Optional[AudioScene] = None,
        transcript: Optional[Transcript] = None,
    ) -> str:
        """
        Answer a question about audio content.

        Args:
            question: Question to answer
            scene: Audio scene (uses current context if None)
            transcript: Transcript (uses current context if None)

        Returns:
            Answer string
        """
        scene = scene or self._current_scene
        transcript = transcript or self._current_transcript

        # Build context
        context = self._build_context(scene, transcript)

        if self.llm:
            # Use LLM for complex reasoning
            return await self._llm_answer(question, context)
        else:
            # Rule-based fallback
            return self._rule_based_answer(question, scene, transcript)

    async def _llm_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        """Answer using LLM."""
        from aion.core.llm import Message

        messages = [
            Message(
                role="system",
                content="""You are an audio analysis assistant. Answer questions about audio content based on the provided analysis.

Be concise and accurate. If you cannot determine something from the context, say so.
Focus on what can be directly observed or inferred from the audio analysis.""",
            ),
            Message(
                role="user",
                content=f"""Audio Analysis:
{context}

Question: {question}

Answer:""",
            ),
        ]

        try:
            response = await self.llm.complete(messages)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM reasoning failed: {e}")
            return "Unable to answer the question."

    def _rule_based_answer(
        self,
        question: str,
        scene: Optional[AudioScene],
        transcript: Optional[Transcript],
    ) -> str:
        """Rule-based answer when LLM is unavailable."""
        q_lower = question.lower()

        if scene:
            # Speaker-related questions
            if "how many" in q_lower and ("speaker" in q_lower or "people" in q_lower):
                return f"There are {len(scene.speakers)} speakers in the audio."

            if "who" in q_lower and "speak" in q_lower:
                if scene.speakers:
                    names = [s.name or f"Speaker {i+1}" for i, s in enumerate(scene.speakers)]
                    return f"The speakers are: {', '.join(names)}"
                return "No speakers were identified."

            # Sound-related questions
            if "what" in q_lower and ("sound" in q_lower or "hear" in q_lower):
                if scene.events:
                    labels = list(set(e.label for e in scene.events))[:5]
                    return f"The following sounds are present: {', '.join(labels)}"
                return "No distinct sounds were detected."

            # Music-related questions
            if "music" in q_lower:
                if scene.music_analysis:
                    m = scene.music_analysis
                    return (
                        f"The music is in {m.key} {m.mode} at approximately "
                        f"{m.tempo_bpm:.0f} BPM. The mood is {m.mood or 'not determined'}."
                    )
                return "No music was detected in the audio."

            # Environment questions
            if "where" in q_lower or "environment" in q_lower or "location" in q_lower:
                return f"Based on the audio, this appears to be {scene.ambient_description or 'an indoor environment'}."

            # Duration questions
            if "how long" in q_lower or "duration" in q_lower:
                return f"The audio is {scene.duration:.1f} seconds long."

        if transcript:
            # Transcript-related questions
            if "what" in q_lower and "say" in q_lower:
                return f"The transcript says: {transcript.text[:500]}..."

            if "language" in q_lower:
                return f"The detected language is {transcript.language}."

        # Default
        if scene:
            return scene.describe()

        return "I don't have enough context to answer that question."

    def _build_context(
        self,
        scene: Optional[AudioScene],
        transcript: Optional[Transcript],
    ) -> str:
        """Build context string for LLM."""
        parts = []

        if scene:
            parts.append(f"Duration: {scene.duration:.1f} seconds")
            parts.append(f"Number of speakers: {len(scene.speakers)}")
            parts.append(f"Number of audio events: {len(scene.events)}")

            if scene.ambient_description:
                parts.append(f"Environment: {scene.ambient_description}")

            if scene.emotional_tone:
                parts.append(f"Emotional tone: {scene.emotional_tone}")

            if scene.events:
                event_labels = list(set(e.label for e in scene.events))[:10]
                parts.append(f"Detected sounds: {', '.join(event_labels)}")

            if scene.music_analysis:
                m = scene.music_analysis
                parts.append(
                    f"Music: {m.key} {m.mode}, {m.tempo_bpm:.0f} BPM, "
                    f"mood: {m.mood or 'unknown'}"
                )

        if transcript:
            parts.append(f"Language: {transcript.language}")
            parts.append(f"Transcript: {transcript.text[:1000]}")

            if transcript.speakers:
                speaker_names = [
                    s.name or f"Speaker {i+1}"
                    for i, s in enumerate(transcript.speakers)
                ]
                parts.append(f"Speakers identified: {', '.join(speaker_names)}")

        return "\n".join(parts)

    async def summarize(
        self,
        scene: Optional[AudioScene] = None,
        transcript: Optional[Transcript] = None,
        max_length: int = 200,
    ) -> str:
        """
        Generate a summary of audio content.

        Args:
            scene: Audio scene
            transcript: Transcript
            max_length: Maximum summary length

        Returns:
            Summary string
        """
        scene = scene or self._current_scene
        transcript = transcript or self._current_transcript

        if self.llm:
            context = self._build_context(scene, transcript)
            return await self._llm_summarize(context, max_length)
        else:
            return self._generate_summary(scene, transcript)

    async def _llm_summarize(self, context: str, max_length: int) -> str:
        """Generate summary using LLM."""
        from aion.core.llm import Message

        messages = [
            Message(
                role="system",
                content="You summarize audio content concisely and accurately.",
            ),
            Message(
                role="user",
                content=f"""Summarize this audio analysis in {max_length} characters or less:

{context}

Summary:""",
            ),
        ]

        try:
            response = await self.llm.complete(messages)
            summary = response.content.strip()
            return summary[:max_length]
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return self._generate_summary(
                self._current_scene, self._current_transcript
            )

    def _generate_summary(
        self,
        scene: Optional[AudioScene],
        transcript: Optional[Transcript],
    ) -> str:
        """Generate summary without LLM."""
        if scene:
            return scene.describe()
        elif transcript:
            return f"Audio transcript ({transcript.duration:.1f}s): {transcript.text[:200]}..."
        return "Audio content processed."

    async def generate_caption(
        self,
        scene: Optional[AudioScene] = None,
        style: str = "descriptive",
    ) -> str:
        """
        Generate a caption for audio content.

        Args:
            scene: Audio scene
            style: Caption style ("descriptive", "brief", "detailed")

        Returns:
            Caption string
        """
        scene = scene or self._current_scene

        if not scene:
            return "Audio"

        if self.llm:
            context = self._build_context(scene, self._current_transcript)

            from aion.core.llm import Message

            style_instructions = {
                "brief": "Generate a very short caption (5-10 words).",
                "descriptive": "Generate a descriptive caption (1-2 sentences).",
                "detailed": "Generate a detailed caption covering all aspects.",
            }

            messages = [
                Message(
                    role="system",
                    content=f"You generate captions for audio content. {style_instructions.get(style, '')}",
                ),
                Message(
                    role="user",
                    content=f"Generate a caption for this audio:\n\n{context}",
                ),
            ]

            try:
                response = await self.llm.complete(messages)
                return response.content.strip()
            except:
                pass

        # Fallback caption generation
        parts = []

        if scene.has_speech:
            speaker_count = len(scene.speakers)
            if speaker_count == 1:
                parts.append("A person speaking")
            else:
                parts.append(f"{speaker_count} people conversing")

        if scene.has_music:
            if scene.music_analysis:
                parts.append(f"{scene.music_analysis.mood or 'background'} music")
            else:
                parts.append("music playing")

        sound_events = [e for e in scene.events if e.category not in ("speech", "music")]
        if sound_events:
            top_sounds = list(set(e.label for e in sound_events[:3]))
            if top_sounds:
                parts.append(f"sounds of {', '.join(top_sounds)}")

        if scene.ambient_description:
            parts.append(f"in {scene.ambient_description}")

        if parts:
            return "; ".join(parts).capitalize()
        return "Audio recording"

    async def chat(
        self,
        user_message: str,
        scene: Optional[AudioScene] = None,
        transcript: Optional[Transcript] = None,
    ) -> str:
        """
        Multi-turn chat about audio content.

        Args:
            user_message: User's message
            scene: Audio scene (uses current context if None)
            transcript: Transcript (uses current context if None)

        Returns:
            Assistant response
        """
        # Update context if provided
        if scene:
            self._current_scene = scene
        if transcript:
            self._current_transcript = transcript

        # Add user message to history
        self._dialogue_history.append(AudioDialogueTurn(
            role="user",
            content=user_message,
        ))

        # Generate response
        if self.llm:
            response = await self._llm_chat(user_message)
        else:
            response = await self.answer_question(user_message)

        # Add assistant response to history
        self._dialogue_history.append(AudioDialogueTurn(
            role="assistant",
            content=response,
        ))

        return response

    async def _llm_chat(self, user_message: str) -> str:
        """Generate chat response using LLM."""
        from aion.core.llm import Message

        # Build context
        context = self._build_context(
            self._current_scene, self._current_transcript
        )

        # Build messages with history
        messages = [
            Message(
                role="system",
                content=f"""You are an audio analysis assistant having a conversation about audio content.

Current audio context:
{context}

Be helpful, concise, and accurate. If you cannot determine something, say so.""",
            ),
        ]

        # Add dialogue history (last 10 turns)
        for turn in self._dialogue_history[-10:]:
            messages.append(Message(
                role=turn.role,
                content=turn.content,
            ))

        try:
            response = await self.llm.complete(messages)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM chat failed: {e}")
            return await self.answer_question(user_message)

    def get_dialogue_history(self) -> list[dict]:
        """Get the dialogue history."""
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat(),
            }
            for turn in self._dialogue_history
        ]


class AudioTranscriptAnalyzer:
    """
    Deep analysis of audio transcripts.

    Provides:
    - Sentiment analysis
    - Topic extraction
    - Key phrase detection
    - Entity recognition
    """

    def __init__(self, llm_adapter=None):
        self.llm = llm_adapter

    async def analyze_sentiment(
        self,
        transcript: Transcript,
    ) -> dict[str, Any]:
        """
        Analyze sentiment of transcript.

        Args:
            transcript: Transcript to analyze

        Returns:
            Sentiment analysis results
        """
        if not transcript.text:
            return {"overall": "neutral", "segments": []}

        # Simple rule-based sentiment
        positive_words = {
            "good", "great", "excellent", "wonderful", "amazing",
            "happy", "love", "best", "fantastic", "awesome",
        }
        negative_words = {
            "bad", "terrible", "awful", "hate", "worst",
            "sad", "angry", "horrible", "disappointing", "poor",
        }

        words = transcript.text.lower().split()
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)

        if positive_count > negative_count * 2:
            overall = "positive"
        elif negative_count > positive_count * 2:
            overall = "negative"
        else:
            overall = "neutral"

        # Per-segment sentiment
        segment_sentiments = []
        for seg in transcript.segments:
            seg_words = seg.text.lower().split()
            pos = sum(1 for w in seg_words if w in positive_words)
            neg = sum(1 for w in seg_words if w in negative_words)

            if pos > neg:
                sentiment = "positive"
            elif neg > pos:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            segment_sentiments.append({
                "segment_id": seg.id,
                "sentiment": sentiment,
                "text_preview": seg.text[:50],
            })

        return {
            "overall": overall,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "segments": segment_sentiments,
        }

    async def extract_topics(
        self,
        transcript: Transcript,
    ) -> list[str]:
        """
        Extract main topics from transcript.

        Args:
            transcript: Transcript to analyze

        Returns:
            List of extracted topics
        """
        if not transcript.text:
            return []

        # Simple keyword-based topic extraction
        topic_keywords = {
            "technology": ["computer", "software", "app", "phone", "internet", "data"],
            "business": ["company", "market", "sales", "revenue", "investment"],
            "health": ["doctor", "medicine", "health", "hospital", "treatment"],
            "education": ["school", "learn", "student", "teacher", "course"],
            "sports": ["game", "team", "player", "score", "win", "match"],
            "politics": ["government", "election", "policy", "law", "vote"],
            "entertainment": ["movie", "music", "show", "actor", "concert"],
            "travel": ["travel", "trip", "vacation", "hotel", "flight"],
        }

        text_lower = transcript.text.lower()
        found_topics = []

        for topic, keywords in topic_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count >= 2:
                found_topics.append(topic)

        return found_topics

    async def extract_key_phrases(
        self,
        transcript: Transcript,
        max_phrases: int = 10,
    ) -> list[str]:
        """
        Extract key phrases from transcript.

        Args:
            transcript: Transcript to analyze
            max_phrases: Maximum phrases to return

        Returns:
            List of key phrases
        """
        if not transcript.text:
            return []

        # Simple n-gram extraction
        words = transcript.text.split()
        phrases = []

        # Extract 2-grams and 3-grams
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                # Filter out phrases with stop words at boundaries
                stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of"}
                if words[i].lower() not in stop_words and words[i + n - 1].lower() not in stop_words:
                    phrases.append(phrase)

        # Count frequency
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # Sort by frequency
        sorted_phrases = sorted(
            phrase_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [p[0] for p in sorted_phrases[:max_phrases]]

    async def extract_entities(
        self,
        transcript: Transcript,
    ) -> dict[str, list[str]]:
        """
        Extract named entities from transcript.

        Args:
            transcript: Transcript to analyze

        Returns:
            Dictionary of entity types to entities
        """
        if not transcript.text:
            return {}

        # Simple pattern-based entity extraction
        text = transcript.text

        entities = {
            "numbers": [],
            "dates": [],
            "times": [],
            "money": [],
        }

        # Extract numbers
        numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
        entities["numbers"] = numbers[:10]

        # Extract date-like patterns
        dates = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?\b',
            text, re.IGNORECASE
        )
        entities["dates"] = dates[:10]

        # Extract time patterns
        times = re.findall(r'\b\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\b', text)
        entities["times"] = times[:10]

        # Extract money patterns
        money = re.findall(r'\$\d+(?:,\d+)*(?:\.\d{2})?', text)
        entities["money"] = money[:10]

        return {k: v for k, v in entities.items() if v}


class AudioReasoningEngine:
    """
    Unified audio reasoning engine.

    Combines:
    - Question answering
    - Summarization
    - Captioning
    - Transcript analysis
    - Multi-turn dialogue
    """

    def __init__(self, llm_adapter=None):
        self.llm = llm_adapter
        self.reasoner = AudioReasoner(llm_adapter)
        self.transcript_analyzer = AudioTranscriptAnalyzer(llm_adapter)

    async def process(
        self,
        scene: AudioScene,
        transcript: Optional[Transcript] = None,
        tasks: list[str] = None,
    ) -> dict[str, Any]:
        """
        Process audio with multiple reasoning tasks.

        Args:
            scene: Audio scene
            transcript: Optional transcript
            tasks: List of tasks ("summarize", "caption", "sentiment", "topics")

        Returns:
            Results for each task
        """
        if tasks is None:
            tasks = ["summarize", "caption"]

        results = {}

        self.reasoner.set_context(scene, transcript)

        if "summarize" in tasks:
            results["summary"] = await self.reasoner.summarize()

        if "caption" in tasks:
            results["caption"] = await self.reasoner.generate_caption(style="descriptive")

        if transcript:
            if "sentiment" in tasks:
                results["sentiment"] = await self.transcript_analyzer.analyze_sentiment(transcript)

            if "topics" in tasks:
                results["topics"] = await self.transcript_analyzer.extract_topics(transcript)

            if "key_phrases" in tasks:
                results["key_phrases"] = await self.transcript_analyzer.extract_key_phrases(transcript)

            if "entities" in tasks:
                results["entities"] = await self.transcript_analyzer.extract_entities(transcript)

        return results

    async def answer(
        self,
        question: str,
        scene: AudioScene,
        transcript: Optional[Transcript] = None,
    ) -> str:
        """Shortcut for answering questions."""
        return await self.reasoner.answer_question(question, scene, transcript)

    async def chat(
        self,
        message: str,
        scene: Optional[AudioScene] = None,
        transcript: Optional[Transcript] = None,
    ) -> str:
        """Shortcut for chat interaction."""
        return await self.reasoner.chat(message, scene, transcript)
