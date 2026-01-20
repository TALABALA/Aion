"""
AION Audio Reasoning

LLM-powered reasoning for audio understanding:
- Question answering about audio content
- Summarization and key point extraction
- Multi-turn conversational analysis
- Cross-modal reasoning with visual context
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioEvent,
    AudioReasoningResult,
    AudioScene,
    Transcript,
    TranscriptSegment,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Prompt Templates
# ============================================================================

AUDIO_QA_SYSTEM_PROMPT = """You are an expert audio analyst. You analyze audio content including:
- Speech transcriptions and conversations
- Audio events and environmental sounds
- Music content and characteristics
- Speaker identification and dynamics

Provide accurate, detailed answers based on the audio information provided.
When uncertain, express your confidence level.
"""

AUDIO_QA_USER_PROMPT = """Based on the following audio analysis, answer the question.

## Audio Scene Information

Duration: {duration:.1f} seconds
Scene Type: {scene_type}
Environment: {ambient_description}

## Speech Content
{transcript_section}

## Detected Events
{events_section}

## Speakers
{speakers_section}

## Music Analysis
{music_section}

## Additional Context
{context}

---

Question: {question}

Provide a clear, accurate answer based on the audio information above.
"""

SUMMARIZATION_PROMPT = """Summarize the following audio content:

## Transcript
{transcript}

## Events Detected
{events}

## Scene Context
{context}

Provide a concise summary covering:
1. Main topics or content
2. Key participants (if speech)
3. Notable events or sounds
4. Overall tone/mood
"""

CONVERSATION_ANALYSIS_PROMPT = """Analyze the following conversation transcript:

{transcript}

Speakers identified: {speakers}

Analyze and provide:
1. Main topics discussed
2. Key points made by each speaker
3. Questions raised and answers given
4. Action items or decisions made
5. Overall tone and dynamics
"""


# ============================================================================
# Audio Reasoning Classes
# ============================================================================

@dataclass
class ReasoningContext:
    """Context for audio reasoning."""
    audio_scene: AudioScene
    additional_context: str = ""
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    visual_context: Optional[dict[str, Any]] = None

    def to_prompt_context(self) -> dict[str, str]:
        """Convert to prompt template variables."""
        # Transcript section
        transcript_section = "No speech detected."
        if self.audio_scene.transcript:
            t = self.audio_scene.transcript
            transcript_section = f"Language: {t.language}\n"
            transcript_section += f"Full text: {t.text}\n"
            if t.segments:
                transcript_section += "\nTimestamped segments:\n"
                for seg in t.segments[:10]:
                    speaker = f"[{seg.speaker_id}] " if seg.speaker_id else ""
                    transcript_section += f"  {speaker}({seg.start_time:.1f}s-{seg.end_time:.1f}s): {seg.text}\n"

        # Events section
        events_section = "No specific events detected."
        if self.audio_scene.events:
            events = []
            for event in self.audio_scene.events[:15]:
                events.append(f"- {event.label} ({event.start_time:.1f}s-{event.end_time:.1f}s, confidence: {event.confidence:.2f})")
            events_section = "\n".join(events)

        # Speakers section
        speakers_section = "No speakers identified."
        if self.audio_scene.speakers:
            speakers = []
            for i, speaker in enumerate(self.audio_scene.speakers):
                name = speaker.name or f"Speaker {i + 1}"
                time = speaker.total_speaking_time
                speakers.append(f"- {name}: {time:.1f}s speaking time")
            speakers_section = "\n".join(speakers)

        # Music section
        music_section = "No music detected."
        if self.audio_scene.has_music and self.audio_scene.music_analysis:
            ma = self.audio_scene.music_analysis
            music_section = f"Tempo: {ma.tempo:.0f} BPM\n"
            music_section += f"Key: {ma.key or 'Unknown'}\n"
            music_section += f"Mood: {ma.mood or 'Unknown'}\n"
            music_section += f"Energy: {ma.energy:.1%}\n"
            if ma.genre:
                music_section += f"Genre: {ma.genre}\n"

        # Context
        context = self.additional_context
        if self.visual_context:
            context += f"\nVisual context: {json.dumps(self.visual_context, default=str)[:500]}"

        return {
            "duration": self.audio_scene.duration,
            "scene_type": self.audio_scene.scene_type,
            "ambient_description": self.audio_scene.ambient_description,
            "transcript_section": transcript_section,
            "events_section": events_section,
            "speakers_section": speakers_section,
            "music_section": music_section,
            "context": context,
        }


class AudioReasoner:
    """
    LLM-powered audio reasoning system.

    Provides question answering, summarization, and analysis
    of audio content using language models.
    """

    def __init__(self, llm_adapter: Optional[Any] = None):
        """
        Initialize audio reasoner.

        Args:
            llm_adapter: LLM adapter for generating responses.
                        Should have an async `generate()` method.
        """
        self._llm = llm_adapter
        self._stats = {
            "questions_answered": 0,
            "summaries_generated": 0,
            "analyses_performed": 0,
        }

    def set_llm(self, llm_adapter: Any) -> None:
        """Set or update the LLM adapter."""
        self._llm = llm_adapter

    async def answer_question(
        self,
        audio_scene: AudioScene,
        question: str,
        additional_context: str = "",
    ) -> AudioReasoningResult:
        """
        Answer a question about audio content.

        Args:
            audio_scene: Analyzed audio scene
            question: Question to answer
            additional_context: Additional context

        Returns:
            AudioReasoningResult
        """
        start_time = time.time()

        # Build context
        context = ReasoningContext(
            audio_scene=audio_scene,
            additional_context=additional_context,
        )
        prompt_vars = context.to_prompt_context()
        prompt_vars["question"] = question

        # Generate answer
        if self._llm:
            answer = await self._generate_llm_answer(prompt_vars)
        else:
            answer = self._generate_heuristic_answer(audio_scene, question)

        # Find relevant segments
        relevant_segments = self._find_relevant_segments(
            audio_scene.transcript, question
        ) if audio_scene.transcript else []

        # Find relevant events
        relevant_events = self._find_relevant_events(audio_scene.events, question)

        # Estimate confidence
        confidence = self._estimate_confidence(audio_scene, question, answer)

        self._stats["questions_answered"] += 1

        return AudioReasoningResult(
            question=question,
            answer=answer,
            confidence=confidence,
            relevant_segments=relevant_segments,
            relevant_events=relevant_events,
            audio_id=audio_scene.audio_id,
            transcript_used=audio_scene.transcript is not None,
            scene_analysis_used=True,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    async def _generate_llm_answer(self, prompt_vars: dict[str, Any]) -> str:
        """Generate answer using LLM."""
        try:
            prompt = AUDIO_QA_USER_PROMPT.format(**prompt_vars)

            response = await self._llm.generate(
                prompt=prompt,
                system_prompt=AUDIO_QA_SYSTEM_PROMPT,
                max_tokens=500,
                temperature=0.3,
            )

            return response.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Unable to generate answer due to an error."

    def _generate_heuristic_answer(
        self,
        audio_scene: AudioScene,
        question: str,
    ) -> str:
        """Generate answer using heuristics when LLM is not available."""
        question_lower = question.lower()

        # Who questions
        if "who" in question_lower:
            if audio_scene.speakers:
                speakers = [s.name or f"Speaker {i+1}"
                           for i, s in enumerate(audio_scene.speakers)]
                return f"The following speakers are identified: {', '.join(speakers)}"
            return "No speakers were identified in the audio."

        # What questions
        if "what" in question_lower:
            if "said" in question_lower or "talk" in question_lower:
                if audio_scene.transcript:
                    return f"The following was said: \"{audio_scene.transcript.text[:500]}...\""
                return "No speech was detected in the audio."

            if "sound" in question_lower or "hear" in question_lower:
                if audio_scene.events:
                    events = [e.label for e in audio_scene.events[:5]]
                    return f"The following sounds were detected: {', '.join(events)}"
                return "No specific sounds were identified."

            if "music" in question_lower:
                if audio_scene.has_music and audio_scene.music_analysis:
                    ma = audio_scene.music_analysis
                    return (f"Music was detected at {ma.tempo:.0f} BPM in {ma.key or 'unknown key'}. "
                           f"The mood is {ma.mood or 'moderate'} with {ma.energy:.0%} energy.")
                return "No music was detected in the audio."

        # When questions
        if "when" in question_lower:
            if audio_scene.events:
                # Find events matching query
                for event in audio_scene.events:
                    if any(word in event.label.lower() for word in question_lower.split()):
                        return f"The '{event.label}' occurs at {event.start_time:.1f} seconds."
            return "Unable to determine specific timing from the audio."

        # How many questions
        if "how many" in question_lower:
            if "speaker" in question_lower:
                return f"There are {len(audio_scene.speakers)} speakers identified."
            if "event" in question_lower or "sound" in question_lower:
                return f"There are {len(audio_scene.events)} audio events detected."
            if "word" in question_lower and audio_scene.transcript:
                return f"The transcript contains approximately {audio_scene.transcript.word_count} words."

        # How long questions
        if "how long" in question_lower:
            return f"The audio is {audio_scene.duration:.1f} seconds ({audio_scene.duration/60:.1f} minutes) long."

        # Default: describe the scene
        return audio_scene.describe()

    def _find_relevant_segments(
        self,
        transcript: Optional[Transcript],
        question: str,
    ) -> list[TranscriptSegment]:
        """Find transcript segments relevant to the question."""
        if not transcript or not transcript.segments:
            return []

        question_words = set(question.lower().split())
        relevant = []

        for segment in transcript.segments:
            segment_words = set(segment.text.lower().split())
            overlap = len(question_words & segment_words)
            if overlap >= 2 or any(len(w) > 5 and w in segment.text.lower()
                                   for w in question_words):
                relevant.append(segment)

        return relevant[:5]

    def _find_relevant_events(
        self,
        events: list[AudioEvent],
        question: str,
    ) -> list[AudioEvent]:
        """Find events relevant to the question."""
        if not events:
            return []

        question_lower = question.lower()
        relevant = []

        for event in events:
            # Check if event label or type matches question
            if event.label.lower() in question_lower:
                relevant.append(event)
            elif event.event_type.value in question_lower:
                relevant.append(event)

        return relevant[:5]

    def _estimate_confidence(
        self,
        audio_scene: AudioScene,
        question: str,
        answer: str,
    ) -> float:
        """Estimate confidence in the answer."""
        confidence = 0.5  # Base confidence

        # Higher confidence if we have transcript for speech questions
        if ("said" in question.lower() or "talk" in question.lower()):
            if audio_scene.transcript and audio_scene.transcript.text:
                confidence += 0.3
            else:
                confidence -= 0.2

        # Higher confidence if we have events for sound questions
        if "sound" in question.lower() or "hear" in question.lower():
            if audio_scene.events:
                confidence += 0.2

        # Higher confidence if we have music analysis for music questions
        if "music" in question.lower():
            if audio_scene.has_music and audio_scene.music_analysis:
                confidence += 0.3

        # Higher confidence if we have speakers for who questions
        if "who" in question.lower():
            if audio_scene.speakers:
                confidence += 0.2

        # Lower confidence for very short answers
        if len(answer) < 50:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    async def summarize(
        self,
        audio_scene: AudioScene,
        focus: Optional[str] = None,
    ) -> str:
        """
        Generate a summary of audio content.

        Args:
            audio_scene: Analyzed audio scene
            focus: Optional focus area ("transcript", "events", "music")

        Returns:
            Summary string
        """
        start_time = time.time()

        if self._llm:
            summary = await self._generate_llm_summary(audio_scene, focus)
        else:
            summary = self._generate_heuristic_summary(audio_scene, focus)

        self._stats["summaries_generated"] += 1

        return summary

    async def _generate_llm_summary(
        self,
        audio_scene: AudioScene,
        focus: Optional[str],
    ) -> str:
        """Generate summary using LLM."""
        try:
            # Build transcript text
            transcript_text = ""
            if audio_scene.transcript:
                if audio_scene.transcript.segments:
                    for seg in audio_scene.transcript.segments:
                        speaker = f"[{seg.speaker_id}] " if seg.speaker_id else ""
                        transcript_text += f"{speaker}{seg.text}\n"
                else:
                    transcript_text = audio_scene.transcript.text

            # Build events text
            events_text = ""
            if audio_scene.events:
                for event in audio_scene.events[:10]:
                    events_text += f"- {event.label} at {event.start_time:.1f}s\n"

            # Build context
            context = f"Duration: {audio_scene.duration:.1f}s\n"
            context += f"Scene type: {audio_scene.scene_type}\n"
            context += f"Environment: {audio_scene.ambient_description}\n"
            if audio_scene.has_music and audio_scene.music_analysis:
                ma = audio_scene.music_analysis
                context += f"Music: {ma.tempo:.0f} BPM, {ma.key or 'unknown key'}, {ma.mood or 'moderate'} mood\n"

            prompt = SUMMARIZATION_PROMPT.format(
                transcript=transcript_text or "No speech detected.",
                events=events_text or "No specific events.",
                context=context,
            )

            if focus:
                prompt += f"\n\nFocus the summary on: {focus}"

            response = await self._llm.generate(
                prompt=prompt,
                system_prompt="You are an expert at summarizing audio content. Be concise but comprehensive.",
                max_tokens=400,
                temperature=0.3,
            )

            return response.strip()

        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return self._generate_heuristic_summary(audio_scene, focus)

    def _generate_heuristic_summary(
        self,
        audio_scene: AudioScene,
        focus: Optional[str],
    ) -> str:
        """Generate summary using heuristics."""
        parts = []

        # Duration and scene type
        parts.append(f"This is a {audio_scene.duration:.1f} second {audio_scene.scene_type} audio recording")
        if audio_scene.ambient_description:
            parts.append(f"in a {audio_scene.ambient_description} environment.")
        else:
            parts.append(".")

        # Transcript summary
        if (focus is None or focus == "transcript") and audio_scene.transcript:
            parts.append(f"Speech detected in {audio_scene.transcript.language}:")
            text = audio_scene.transcript.text
            if len(text) > 200:
                parts.append(f'"{text[:200]}..."')
            else:
                parts.append(f'"{text}"')

        # Speakers summary
        if audio_scene.speakers:
            if len(audio_scene.speakers) == 1:
                parts.append("One speaker was identified.")
            else:
                parts.append(f"{len(audio_scene.speakers)} speakers were identified.")

        # Events summary
        if (focus is None or focus == "events") and audio_scene.events:
            event_labels = list(set(e.label for e in audio_scene.events))[:5]
            parts.append(f"Detected sounds include: {', '.join(event_labels)}.")

        # Music summary
        if (focus is None or focus == "music") and audio_scene.has_music and audio_scene.music_analysis:
            ma = audio_scene.music_analysis
            parts.append(f"Music detected at {ma.tempo:.0f} BPM in {ma.key or 'unknown key'} "
                        f"with a {ma.mood or 'moderate'} mood.")

        return " ".join(parts)

    async def analyze_conversation(
        self,
        transcript: Transcript,
    ) -> dict[str, Any]:
        """
        Analyze a conversation from transcript.

        Args:
            transcript: Transcript with speaker information

        Returns:
            Analysis results
        """
        start_time = time.time()

        if self._llm:
            analysis = await self._generate_llm_conversation_analysis(transcript)
        else:
            analysis = self._generate_heuristic_conversation_analysis(transcript)

        self._stats["analyses_performed"] += 1

        return analysis

    async def _generate_llm_conversation_analysis(
        self,
        transcript: Transcript,
    ) -> dict[str, Any]:
        """Generate conversation analysis using LLM."""
        try:
            # Build transcript text with speakers
            transcript_text = ""
            for seg in transcript.segments:
                speaker = seg.speaker_id or "Unknown"
                transcript_text += f"[{speaker}]: {seg.text}\n"

            speakers = [s.name or s.id for s in transcript.speakers]

            prompt = CONVERSATION_ANALYSIS_PROMPT.format(
                transcript=transcript_text,
                speakers=", ".join(speakers) if speakers else "Unknown",
            )

            response = await self._llm.generate(
                prompt=prompt,
                system_prompt="You are an expert conversation analyst. Provide structured analysis.",
                max_tokens=600,
                temperature=0.3,
            )

            # Parse response into structured format
            return {
                "analysis": response.strip(),
                "speaker_count": len(transcript.speakers),
                "turn_count": len(transcript.segments),
                "duration": transcript.duration,
                "word_count": transcript.word_count,
            }

        except Exception as e:
            logger.error(f"LLM conversation analysis failed: {e}")
            return self._generate_heuristic_conversation_analysis(transcript)

    def _generate_heuristic_conversation_analysis(
        self,
        transcript: Transcript,
    ) -> dict[str, Any]:
        """Generate conversation analysis using heuristics."""
        # Count turns per speaker
        speaker_turns: dict[str, int] = {}
        speaker_words: dict[str, int] = {}

        for seg in transcript.segments:
            speaker = seg.speaker_id or "Unknown"
            speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1
            speaker_words[speaker] = speaker_words.get(speaker, 0) + len(seg.text.split())

        # Find questions
        questions = [seg.text for seg in transcript.segments
                    if seg.text.strip().endswith("?")]

        # Calculate speaking balance
        total_words = sum(speaker_words.values())
        balance = {}
        for speaker, words in speaker_words.items():
            balance[speaker] = words / total_words if total_words > 0 else 0

        return {
            "speaker_count": len(transcript.speakers),
            "turn_count": len(transcript.segments),
            "duration": transcript.duration,
            "word_count": transcript.word_count,
            "speaker_turns": speaker_turns,
            "speaker_words": speaker_words,
            "speaking_balance": balance,
            "questions_count": len(questions),
            "questions": questions[:5],  # First 5 questions
        }

    def get_stats(self) -> dict[str, Any]:
        """Get reasoning statistics."""
        return {
            **self._stats,
            "llm_available": self._llm is not None,
        }


# ============================================================================
# Multi-Modal Reasoner
# ============================================================================

class MultiModalReasoner:
    """
    Reasoner that combines audio and visual understanding.

    Enables questions that span both modalities:
    - "Who is talking and what are they doing?"
    - "Does the sound match what's happening visually?"
    - "What emotions are expressed verbally vs facially?"
    """

    def __init__(self, llm_adapter: Optional[Any] = None):
        """Initialize multi-modal reasoner."""
        self._llm = llm_adapter
        self._audio_reasoner = AudioReasoner(llm_adapter)

    async def answer_multimodal_question(
        self,
        audio_scene: AudioScene,
        visual_scene: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """
        Answer a question using both audio and visual context.

        Args:
            audio_scene: Audio scene analysis
            visual_scene: Visual scene analysis
            question: Question to answer

        Returns:
            Answer with supporting evidence from both modalities
        """
        # Build combined context
        context = self._build_multimodal_context(audio_scene, visual_scene)

        # Get audio-based answer
        audio_result = await self._audio_reasoner.answer_question(
            audio_scene, question, context
        )

        # Determine if question needs visual context
        needs_visual = self._needs_visual_context(question)

        if needs_visual and visual_scene:
            # Enhance answer with visual information
            visual_evidence = self._extract_visual_evidence(visual_scene, question)
            return {
                "answer": audio_result.answer,
                "audio_evidence": {
                    "relevant_segments": [s.to_dict() for s in audio_result.relevant_segments],
                    "relevant_events": [e.to_dict() for e in audio_result.relevant_events],
                },
                "visual_evidence": visual_evidence,
                "modalities_used": ["audio", "visual"],
                "confidence": audio_result.confidence,
            }

        return {
            "answer": audio_result.answer,
            "audio_evidence": {
                "relevant_segments": [s.to_dict() for s in audio_result.relevant_segments],
                "relevant_events": [e.to_dict() for e in audio_result.relevant_events],
            },
            "visual_evidence": None,
            "modalities_used": ["audio"],
            "confidence": audio_result.confidence,
        }

    def _build_multimodal_context(
        self,
        audio_scene: AudioScene,
        visual_scene: dict[str, Any],
    ) -> str:
        """Build context string from both modalities."""
        context_parts = []

        # Visual context
        if visual_scene:
            if "caption" in visual_scene:
                context_parts.append(f"Visual scene: {visual_scene['caption']}")
            if "objects" in visual_scene:
                objects = [o.get("label", "unknown") for o in visual_scene["objects"][:5]]
                context_parts.append(f"Visible objects: {', '.join(objects)}")

        return " | ".join(context_parts)

    def _needs_visual_context(self, question: str) -> bool:
        """Determine if question needs visual context."""
        visual_keywords = [
            "see", "look", "visual", "show", "appear",
            "face", "expression", "gesture", "movement",
            "doing", "action", "activity",
        ]
        question_lower = question.lower()
        return any(kw in question_lower for kw in visual_keywords)

    def _extract_visual_evidence(
        self,
        visual_scene: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """Extract relevant visual evidence for the question."""
        evidence = {}

        if "objects" in visual_scene:
            # Filter objects relevant to question
            question_words = set(question.lower().split())
            relevant_objects = []
            for obj in visual_scene["objects"]:
                label = obj.get("label", "").lower()
                if any(word in label for word in question_words):
                    relevant_objects.append(obj)
            evidence["relevant_objects"] = relevant_objects[:5]

        if "caption" in visual_scene:
            evidence["scene_description"] = visual_scene["caption"]

        if "relations" in visual_scene:
            evidence["relations"] = visual_scene["relations"][:5]

        return evidence
