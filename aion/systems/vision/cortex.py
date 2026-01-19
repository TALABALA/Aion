"""
AION Visual Cortex

Unified visual processing system that integrates:
- Visual perception (detection, captioning, segmentation)
- Visual memory (storage and retrieval)
- Visual reasoning (scene understanding, VQA)
- Attention mechanisms
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np
import structlog

from aion.systems.vision.perception import (
    VisualPerception,
    DetectedObject,
    SceneGraph,
    BoundingBox,
)
from aion.systems.vision.memory import VisualMemory, VisualMemoryEntry

logger = structlog.get_logger(__name__)


@dataclass
class VisualAttention:
    """Attention focus on a region of an image."""
    region: BoundingBox
    focus_type: str  # "object", "region", "global"
    target_id: Optional[str] = None
    confidence: float = 1.0


@dataclass
class VisualAnalysisResult:
    """Complete result of visual analysis."""
    scene_graph: SceneGraph
    caption: str
    attention: list[VisualAttention]
    reasoning: Optional[str] = None
    similar_memories: list[VisualMemoryEntry] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_graph": self.scene_graph.to_dict(),
            "caption": self.caption,
            "attention": [
                {
                    "region": a.region.to_dict(),
                    "focus_type": a.focus_type,
                    "target_id": a.target_id,
                    "confidence": a.confidence,
                }
                for a in self.attention
            ],
            "reasoning": self.reasoning,
            "similar_memories_count": len(self.similar_memories),
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


class VisualCortex:
    """
    AION Visual Cortex

    The unified visual processing center that coordinates:
    - Low-level perception (object detection, segmentation)
    - Mid-level understanding (scene graphs, relationships)
    - High-level reasoning (VQA, scene description)
    - Memory integration (similar scene retrieval)
    """

    def __init__(
        self,
        detection_model: str = "facebook/detr-resnet-50",
        captioning_model: str = "Salesforce/blip-image-captioning-base",
        enable_memory: bool = True,
        attention_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.detection_model = detection_model
        self.captioning_model = captioning_model
        self.enable_memory = enable_memory
        self.attention_threshold = attention_threshold
        self.device = device

        # Components
        self._perception = VisualPerception(
            detection_model=detection_model,
            captioning_model=captioning_model,
            device=device,
        )
        self._memory = VisualMemory() if enable_memory else None

        # Statistics
        self._stats = {
            "images_processed": 0,
            "questions_answered": 0,
            "total_processing_time_ms": 0.0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the visual cortex."""
        if self._initialized:
            return

        logger.info("Initializing Visual Cortex")

        await self._perception.initialize()
        if self._memory:
            await self._memory.initialize()

        self._initialized = True
        logger.info("Visual Cortex initialized")

    async def shutdown(self) -> None:
        """Shutdown the visual cortex."""
        await self._perception.shutdown()
        if self._memory:
            await self._memory.shutdown()
        self._initialized = False

    async def process(
        self,
        image_path: Union[str, Path],
        query: Optional[str] = None,
        store_in_memory: bool = True,
    ) -> VisualAnalysisResult:
        """
        Process an image with full visual analysis.

        Args:
            image_path: Path or URL to the image
            query: Optional question about the image
            store_in_memory: Whether to store in visual memory

        Returns:
            VisualAnalysisResult with complete analysis
        """
        if not self._initialized:
            await self.initialize()

        import time
        start_time = time.monotonic()

        # Build scene graph (includes detection)
        scene_graph = await self._perception.build_scene_graph(image_path)

        # Generate caption if not in scene graph
        caption = scene_graph.global_features.get("caption", "")
        if not caption:
            caption = await self._perception.generate_caption(image_path)

        # Generate attention map
        attention = self._compute_attention(scene_graph)

        # Answer question if provided
        reasoning = None
        if query:
            reasoning = await self._perception.answer_question(image_path, query)
            self._stats["questions_answered"] += 1

        # Search for similar memories
        similar_memories = []
        if self._memory and self._memory.count() > 0:
            results = await self._memory.search_by_similarity(scene_graph, limit=3)
            similar_memories = [r.entry for r in results]

        # Store in memory
        if store_in_memory and self._memory:
            await self._memory.store(
                scene_graph=scene_graph,
                metadata={
                    "source": str(image_path),
                    "query": query,
                    "reasoning": reasoning,
                },
            )

        processing_time_ms = (time.monotonic() - start_time) * 1000

        self._stats["images_processed"] += 1
        self._stats["total_processing_time_ms"] += processing_time_ms

        return VisualAnalysisResult(
            scene_graph=scene_graph,
            caption=caption,
            attention=attention,
            reasoning=reasoning,
            similar_memories=similar_memories,
            processing_time_ms=processing_time_ms,
            metadata={
                "source": str(image_path),
                "query": query,
            },
        )

    def _compute_attention(
        self,
        scene_graph: SceneGraph,
    ) -> list[VisualAttention]:
        """
        Compute attention focus areas for a scene.

        Uses object confidence and size to determine attention.
        """
        attention = []

        # Sort objects by importance (confidence * size)
        scored_objects = []
        for obj in scene_graph.objects:
            area = obj.bounding_box.area
            score = obj.confidence * (1 + np.log1p(area * 100))
            scored_objects.append((obj, score))

        scored_objects.sort(key=lambda x: x[1], reverse=True)

        # Create attention for top objects
        for obj, score in scored_objects[:5]:
            if obj.confidence >= self.attention_threshold:
                attention.append(VisualAttention(
                    region=obj.bounding_box,
                    focus_type="object",
                    target_id=obj.id,
                    confidence=obj.confidence,
                ))

        # Add global attention if no strong focus
        if not attention:
            attention.append(VisualAttention(
                region=BoundingBox(0, 0, 1, 1),
                focus_type="global",
                confidence=0.5,
            ))

        return attention

    async def detect_objects(
        self,
        image: Union[str, Path],
        threshold: float = 0.5,
    ) -> list[DetectedObject]:
        """Detect objects in an image."""
        if not self._initialized:
            await self.initialize()
        return await self._perception.detect_objects(image, threshold)

    async def describe(
        self,
        image: Union[str, Path],
    ) -> str:
        """Generate a natural language description of an image."""
        if not self._initialized:
            await self.initialize()

        scene_graph = await self._perception.build_scene_graph(image)
        return scene_graph.describe()

    async def answer(
        self,
        image: Union[str, Path],
        question: str,
    ) -> str:
        """Answer a question about an image."""
        if not self._initialized:
            await self.initialize()

        self._stats["questions_answered"] += 1
        return await self._perception.answer_question(image, question)

    async def compare(
        self,
        image1: Union[str, Path],
        image2: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Compare two images.

        Returns:
            Comparison results including similarity and differences
        """
        if not self._initialized:
            await self.initialize()

        # Analyze both images
        scene1 = await self._perception.build_scene_graph(image1)
        scene2 = await self._perception.build_scene_graph(image2)

        # Compare objects
        labels1 = {obj.label for obj in scene1.objects}
        labels2 = {obj.label for obj in scene2.objects}

        common = labels1 & labels2
        only_in_1 = labels1 - labels2
        only_in_2 = labels2 - labels1

        # Compute similarity
        similarity = len(common) / max(len(labels1 | labels2), 1)

        return {
            "similarity": similarity,
            "common_objects": list(common),
            "unique_to_image1": list(only_in_1),
            "unique_to_image2": list(only_in_2),
            "image1_caption": scene1.global_features.get("caption", ""),
            "image2_caption": scene2.global_features.get("caption", ""),
        }

    async def recall_similar(
        self,
        image: Union[str, Path],
        limit: int = 5,
    ) -> list[VisualMemoryEntry]:
        """
        Recall similar images from visual memory.

        Args:
            image: Query image
            limit: Maximum results

        Returns:
            List of similar memory entries
        """
        if not self._memory:
            return []

        if not self._initialized:
            await self.initialize()

        scene_graph = await self._perception.build_scene_graph(image)
        results = await self._memory.search_by_similarity(scene_graph, limit)
        return [r.entry for r in results]

    async def imagine(
        self,
        description: str,
    ) -> SceneGraph:
        """
        Imagine a scene from a text description.

        Creates a synthetic scene graph based on the description.
        This is a simplified implementation - in production, would
        use a generative model.

        Args:
            description: Text description of the scene

        Returns:
            Imagined SceneGraph
        """
        # Parse description for objects
        common_objects = [
            "person", "car", "dog", "cat", "chair", "table",
            "tree", "building", "phone", "computer", "book",
        ]

        description_lower = description.lower()
        detected_objects = []

        for i, obj_label in enumerate(common_objects):
            if obj_label in description_lower:
                # Create imagined object at random position
                x = 0.2 + (i % 3) * 0.3
                y = 0.3 + (i // 3) * 0.3

                detected_objects.append(DetectedObject(
                    id=f"imagined_{i}",
                    label=obj_label,
                    confidence=0.7,
                    bounding_box=BoundingBox(
                        x_min=x - 0.1,
                        y_min=y - 0.1,
                        x_max=x + 0.1,
                        y_max=y + 0.1,
                    ),
                    attributes={"imagined": True},
                ))

        # Create scene graph
        scene_graph = SceneGraph(
            objects=detected_objects,
            relations=[],
            global_features={
                "caption": description,
                "imagined": True,
            },
        )

        # Infer relations
        scene_graph.relations = self._perception._infer_spatial_relations(
            detected_objects
        )

        return scene_graph

    def get_stats(self) -> dict[str, Any]:
        """Get visual cortex statistics."""
        stats = {**self._stats}

        if stats["images_processed"] > 0:
            stats["avg_processing_time_ms"] = (
                stats["total_processing_time_ms"] / stats["images_processed"]
            )

        if self._memory:
            stats["memory"] = self._memory.get_stats()

        return stats

    @property
    def memory(self) -> Optional[VisualMemory]:
        """Get the visual memory system."""
        return self._memory
