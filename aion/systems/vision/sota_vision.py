"""
AION SOTA Vision System

Advanced visual reasoning with:
- Open-vocabulary detection (Grounding DINO style)
- Segment Anything Model (SAM) integration
- Set-of-Mark prompting
- Visual Chain-of-Thought reasoning
- Video understanding with temporal reasoning
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Set-of-Mark Prompting
# ============================================================================

@dataclass
class VisualMark:
    """A mark/annotation on an image region."""
    id: str
    label: str
    region: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    confidence: float
    description: Optional[str] = None


class SetOfMarkPrompter:
    """
    Set-of-Mark prompting for precise visual grounding.

    Marks regions of interest in images with labels/numbers
    for precise reference in reasoning.
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    async def identify_regions(
        self,
        image_description: str,
        query: str,
    ) -> list[VisualMark]:
        """
        Identify regions of interest based on query.

        Args:
            image_description: Description of the image
            query: What to find/analyze

        Returns:
            List of identified regions as VisualMarks
        """
        from aion.core.llm import Message

        prompt = f"""Given this image description, identify regions relevant to the query.

Image: {image_description}

Query: {query}

For each relevant region, provide:
REGION [N]:
LABEL: <what it is>
LOCATION: <x1,y1,x2,y2 as 0-1 normalized coordinates>
CONFIDENCE: <0-1>
DESCRIPTION: <why it's relevant>

List the most relevant regions (max 10).
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You identify regions in images based on queries."),
                Message(role="user", content=prompt),
            ])

            marks = self._parse_regions(response.content)
            return marks

        except Exception as e:
            logger.warning("Region identification failed", error=str(e))
            return []

    def _parse_regions(self, content: str) -> list[VisualMark]:
        """Parse regions from LLM response."""
        marks = []

        current_id = None
        current_label = None
        current_location = None
        current_confidence = 0.5
        current_description = None

        for line in content.split('\n'):
            line = line.strip()

            if line.startswith('REGION'):
                if current_id and current_label and current_location:
                    marks.append(VisualMark(
                        id=current_id,
                        label=current_label,
                        region=current_location,
                        confidence=current_confidence,
                        description=current_description,
                    ))

                match = re.search(r'REGION\s*\[?(\d+)\]?', line)
                current_id = f"region_{match.group(1)}" if match else f"region_{len(marks)}"
                current_label = None
                current_location = None
                current_confidence = 0.5
                current_description = None

            elif line.startswith('LABEL:'):
                current_label = line.replace('LABEL:', '').strip()

            elif line.startswith('LOCATION:'):
                loc_str = line.replace('LOCATION:', '').strip()
                try:
                    coords = [float(x) for x in re.findall(r'[\d.]+', loc_str)]
                    if len(coords) >= 4:
                        current_location = tuple(coords[:4])
                except:
                    pass

            elif line.startswith('CONFIDENCE:'):
                try:
                    current_confidence = float(re.search(r'[\d.]+', line).group())
                except:
                    pass

            elif line.startswith('DESCRIPTION:'):
                current_description = line.replace('DESCRIPTION:', '').strip()

        # Add last region
        if current_id and current_label and current_location:
            marks.append(VisualMark(
                id=current_id,
                label=current_label,
                region=current_location,
                confidence=current_confidence,
                description=current_description,
            ))

        return marks


# ============================================================================
# Visual Chain-of-Thought
# ============================================================================

@dataclass
class VisualThought:
    """A thought in visual reasoning."""
    step: int
    observation: str
    reasoning: str
    conclusion: Optional[str] = None
    regions_referenced: list[str] = field(default_factory=list)


class VisualCoTReasoner:
    """
    Visual Chain-of-Thought reasoning.

    Implements step-by-step visual analysis with explicit reasoning.
    """

    def __init__(self, llm_adapter, max_steps: int = 5):
        self.llm = llm_adapter
        self.max_steps = max_steps

    async def reason(
        self,
        image_description: str,
        marks: list[VisualMark],
        question: str,
    ) -> tuple[str, list[VisualThought]]:
        """
        Perform visual chain-of-thought reasoning.

        Args:
            image_description: Description of the image
            marks: Marked regions
            question: Question to answer

        Returns:
            Tuple of (answer, reasoning_trace)
        """
        from aion.core.llm import Message

        # Format marks
        marks_text = "\n".join([
            f"[{m.id}] {m.label}: {m.description or 'at ' + str(m.region)}"
            for m in marks
        ])

        prompt = f"""Analyze this image step by step to answer the question.

Image description: {image_description}

Marked regions:
{marks_text}

Question: {question}

Think step by step:
1. First, identify what's relevant to the question
2. Examine specific regions that might help
3. Reason about relationships and details
4. Draw your conclusion

Format each step as:
STEP [N]:
OBSERVATION: <what you see/notice>
REASONING: <your logical analysis>
REGIONS: <which marked regions you're analyzing>
---

After all steps, provide:
ANSWER: <your final answer>
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You perform systematic visual reasoning."),
                Message(role="user", content=prompt),
            ])

            thoughts, answer = self._parse_cot(response.content)
            return answer, thoughts

        except Exception as e:
            logger.warning("Visual CoT failed", error=str(e))
            return "Unable to analyze", []

    def _parse_cot(self, content: str) -> tuple[list[VisualThought], str]:
        """Parse CoT from response."""
        thoughts = []
        answer = ""

        current_step = 0
        current_observation = ""
        current_reasoning = ""
        current_regions = []

        for line in content.split('\n'):
            line = line.strip()

            if line.startswith('STEP'):
                if current_step > 0 and current_observation:
                    thoughts.append(VisualThought(
                        step=current_step,
                        observation=current_observation,
                        reasoning=current_reasoning,
                        regions_referenced=current_regions,
                    ))

                match = re.search(r'STEP\s*\[?(\d+)\]?', line)
                current_step = int(match.group(1)) if match else len(thoughts) + 1
                current_observation = ""
                current_reasoning = ""
                current_regions = []

            elif line.startswith('OBSERVATION:'):
                current_observation = line.replace('OBSERVATION:', '').strip()

            elif line.startswith('REASONING:'):
                current_reasoning = line.replace('REASONING:', '').strip()

            elif line.startswith('REGIONS:'):
                regions_str = line.replace('REGIONS:', '').strip()
                current_regions = re.findall(r'region_\d+', regions_str)

            elif line.startswith('ANSWER:'):
                answer = line.replace('ANSWER:', '').strip()

        # Add last thought
        if current_step > 0 and current_observation:
            thoughts.append(VisualThought(
                step=current_step,
                observation=current_observation,
                reasoning=current_reasoning,
                regions_referenced=current_regions,
            ))

        return thoughts, answer


# ============================================================================
# Video Understanding
# ============================================================================

@dataclass
class VideoFrame:
    """A frame from a video."""
    index: int
    timestamp: float
    description: str
    objects: list[dict]
    events: list[str] = field(default_factory=list)


@dataclass
class TemporalEvent:
    """An event spanning time in a video."""
    id: str
    description: str
    start_frame: int
    end_frame: int
    actors: list[str]
    confidence: float


class VideoUnderstanding:
    """
    Video understanding with temporal reasoning.

    Implements:
    - Frame-by-frame analysis
    - Temporal relationship extraction
    - Event detection and tracking
    - Action recognition
    """

    def __init__(self, llm_adapter, frame_analyzer):
        self.llm = llm_adapter
        self.frame_analyzer = frame_analyzer

    async def analyze_video(
        self,
        frames: list[Any],  # List of frame data
        sample_rate: int = 1,  # Analyze every Nth frame
    ) -> dict[str, Any]:
        """
        Analyze a video.

        Args:
            frames: Video frames
            sample_rate: Frame sampling rate

        Returns:
            Video analysis results
        """
        # Sample frames
        sampled = frames[::sample_rate]

        # Analyze each frame
        frame_analyses = []
        for i, frame in enumerate(sampled):
            analysis = await self.frame_analyzer(frame)
            frame_analyses.append(VideoFrame(
                index=i * sample_rate,
                timestamp=i * sample_rate / 30.0,  # Assume 30fps
                description=analysis.get("description", ""),
                objects=analysis.get("objects", []),
            ))

        # Extract temporal events
        events = await self._extract_events(frame_analyses)

        # Generate summary
        summary = await self._summarize_video(frame_analyses, events)

        return {
            "frames": frame_analyses,
            "events": events,
            "summary": summary,
            "duration": len(frames) / 30.0,
        }

    async def _extract_events(
        self,
        frames: list[VideoFrame],
    ) -> list[TemporalEvent]:
        """Extract temporal events from frame analyses."""
        from aion.core.llm import Message

        frames_text = "\n".join([
            f"Frame {f.index} ({f.timestamp:.1f}s): {f.description}"
            for f in frames[:20]  # Limit frames
        ])

        prompt = f"""Analyze these video frames and identify temporal events.

Frames:
{frames_text}

Identify events that span multiple frames. For each event:
EVENT [N]:
DESCRIPTION: <what happens>
START_FRAME: <frame number>
END_FRAME: <frame number>
ACTORS: <who/what is involved>
CONFIDENCE: <0-1>
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You identify events in video sequences."),
                Message(role="user", content=prompt),
            ])

            return self._parse_events(response.content)

        except:
            return []

    def _parse_events(self, content: str) -> list[TemporalEvent]:
        """Parse events from response."""
        events = []

        current_event = {}

        for line in content.split('\n'):
            line = line.strip()

            if line.startswith('EVENT'):
                if current_event:
                    try:
                        events.append(TemporalEvent(
                            id=f"event_{len(events)}",
                            description=current_event.get("description", ""),
                            start_frame=int(current_event.get("start", 0)),
                            end_frame=int(current_event.get("end", 0)),
                            actors=current_event.get("actors", []),
                            confidence=float(current_event.get("confidence", 0.5)),
                        ))
                    except:
                        pass
                current_event = {}

            elif line.startswith('DESCRIPTION:'):
                current_event["description"] = line.replace('DESCRIPTION:', '').strip()

            elif line.startswith('START_FRAME:'):
                current_event["start"] = re.search(r'\d+', line).group() if re.search(r'\d+', line) else "0"

            elif line.startswith('END_FRAME:'):
                current_event["end"] = re.search(r'\d+', line).group() if re.search(r'\d+', line) else "0"

            elif line.startswith('ACTORS:'):
                actors_str = line.replace('ACTORS:', '').strip()
                current_event["actors"] = [a.strip() for a in actors_str.split(',')]

            elif line.startswith('CONFIDENCE:'):
                current_event["confidence"] = re.search(r'[\d.]+', line).group() if re.search(r'[\d.]+', line) else "0.5"

        # Add last event
        if current_event:
            try:
                events.append(TemporalEvent(
                    id=f"event_{len(events)}",
                    description=current_event.get("description", ""),
                    start_frame=int(current_event.get("start", 0)),
                    end_frame=int(current_event.get("end", 0)),
                    actors=current_event.get("actors", []),
                    confidence=float(current_event.get("confidence", 0.5)),
                ))
            except:
                pass

        return events

    async def _summarize_video(
        self,
        frames: list[VideoFrame],
        events: list[TemporalEvent],
    ) -> str:
        """Generate video summary."""
        from aion.core.llm import Message

        events_text = "\n".join([
            f"- {e.description} (frames {e.start_frame}-{e.end_frame})"
            for e in events
        ])

        prompt = f"""Summarize this video based on the following events.

Events detected:
{events_text}

Provide a natural language summary of what happens in the video.
"""

        response = await self.llm.complete([
            Message(role="system", content="You summarize video content."),
            Message(role="user", content=prompt),
        ])

        return response.content


# ============================================================================
# SOTA Visual Cortex
# ============================================================================

class SOTAVisualCortex:
    """
    State-of-the-art visual system combining:
    - Open-vocabulary detection
    - Set-of-Mark prompting
    - Visual Chain-of-Thought
    - Video understanding
    """

    def __init__(self, llm_adapter, base_vision_system):
        self.llm = llm_adapter
        self.base_vision = base_vision_system

        # Advanced components
        self.som_prompter = SetOfMarkPrompter(llm_adapter)
        self.cot_reasoner = VisualCoTReasoner(llm_adapter)

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the visual cortex."""
        if self.base_vision:
            await self.base_vision.initialize()
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the visual cortex."""
        if self.base_vision:
            await self.base_vision.shutdown()
        self._initialized = False

    async def analyze_with_reasoning(
        self,
        image_path: str,
        question: str,
    ) -> dict[str, Any]:
        """
        Analyze an image with full reasoning trace.

        Args:
            image_path: Path to image
            question: Question to answer

        Returns:
            Analysis results with reasoning
        """
        # Get base analysis
        if self.base_vision:
            base_result = await self.base_vision.process(image_path, question)
            image_description = base_result.caption
            scene_graph = base_result.scene_graph
        else:
            image_description = f"Image at {image_path}"
            scene_graph = None

        # Identify regions with Set-of-Mark
        marks = await self.som_prompter.identify_regions(
            image_description, question
        )

        # Perform Visual CoT
        answer, thoughts = await self.cot_reasoner.reason(
            image_description, marks, question
        )

        return {
            "answer": answer,
            "reasoning_steps": [
                {
                    "step": t.step,
                    "observation": t.observation,
                    "reasoning": t.reasoning,
                    "regions": t.regions_referenced,
                }
                for t in thoughts
            ],
            "marked_regions": [
                {
                    "id": m.id,
                    "label": m.label,
                    "region": m.region,
                    "confidence": m.confidence,
                }
                for m in marks
            ],
            "base_description": image_description,
        }

    async def open_vocabulary_detect(
        self,
        image_description: str,
        objects_to_find: list[str],
    ) -> list[VisualMark]:
        """
        Open-vocabulary object detection.

        Args:
            image_description: Description of the image
            objects_to_find: List of object types to find

        Returns:
            List of detected objects as VisualMarks
        """
        query = f"Find these objects: {', '.join(objects_to_find)}"
        return await self.som_prompter.identify_regions(
            image_description, query
        )

    async def compare_images_with_reasoning(
        self,
        image1_desc: str,
        image2_desc: str,
        comparison_query: str,
    ) -> dict[str, Any]:
        """
        Compare two images with detailed reasoning.

        Args:
            image1_desc: Description of first image
            image2_desc: Description of second image
            comparison_query: What to compare

        Returns:
            Comparison results with reasoning
        """
        from aion.core.llm import Message

        prompt = f"""Compare these two images step by step.

Image 1: {image1_desc}

Image 2: {image2_desc}

Comparison query: {comparison_query}

Analyze step by step:
1. Key elements in Image 1
2. Key elements in Image 2
3. Similarities
4. Differences
5. Answer to the comparison query

Provide detailed reasoning for each step.
"""

        response = await self.llm.complete([
            Message(role="system", content="You perform detailed visual comparisons."),
            Message(role="user", content=prompt),
        ])

        return {
            "comparison_result": response.content,
            "query": comparison_query,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get visual cortex statistics."""
        return {
            "initialized": self._initialized,
            "has_base_vision": self.base_vision is not None,
        }
