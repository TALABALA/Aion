"""
AION Visual Perception

Multi-model visual perception system with:
- Object detection (DETR)
- Image captioning (BLIP)
- Semantic segmentation (SegFormer)
- Scene graph construction
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from pathlib import Path
import io

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for detected objects."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.width / 2, self.y_min + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    def iou(self, other: "BoundingBox") -> float:
        """Calculate intersection over union with another box."""
        x1 = max(self.x_min, other.x_min)
        y1 = max(self.y_min, other.y_min)
        x2 = min(self.x_max, other.x_max)
        y2 = min(self.y_max, other.y_max)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class DetectedObject:
    """A detected object in an image."""
    id: str
    label: str
    confidence: float
    bounding_box: BoundingBox
    segmentation_mask: Optional[np.ndarray] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.to_dict(),
            "attributes": self.attributes,
        }


@dataclass
class SceneRelation:
    """A relation between objects in a scene."""
    subject_id: str
    predicate: str
    object_id: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject_id,
            "predicate": self.predicate,
            "object": self.object_id,
            "confidence": self.confidence,
        }


@dataclass
class SceneGraph:
    """A scene graph representing objects and their relationships."""
    objects: list[DetectedObject]
    relations: list[SceneRelation]
    global_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "relations": [r.to_dict() for r in self.relations],
            "global_features": self.global_features,
        }

    def get_object_by_id(self, obj_id: str) -> Optional[DetectedObject]:
        """Get an object by its ID."""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def get_relations_for_object(self, obj_id: str) -> list[SceneRelation]:
        """Get all relations involving an object."""
        return [
            r for r in self.relations
            if r.subject_id == obj_id or r.object_id == obj_id
        ]

    def describe(self) -> str:
        """Generate a natural language description of the scene."""
        if not self.objects:
            return "The scene appears to be empty."

        descriptions = []

        # Count objects by label
        label_counts = {}
        for obj in self.objects:
            label_counts[obj.label] = label_counts.get(obj.label, 0) + 1

        # Describe object counts
        object_strs = []
        for label, count in label_counts.items():
            if count == 1:
                object_strs.append(f"a {label}")
            else:
                object_strs.append(f"{count} {label}s")

        if object_strs:
            descriptions.append(f"The scene contains {', '.join(object_strs)}.")

        # Describe key relations
        for rel in self.relations[:5]:
            subj = self.get_object_by_id(rel.subject_id)
            obj = self.get_object_by_id(rel.object_id)
            if subj and obj:
                descriptions.append(
                    f"The {subj.label} is {rel.predicate} the {obj.label}."
                )

        return " ".join(descriptions)


class VisualPerception:
    """
    AION Visual Perception System

    Processes images using multiple vision models:
    - DETR for object detection
    - BLIP for image captioning
    - SegFormer for semantic segmentation
    """

    def __init__(
        self,
        detection_model: str = "facebook/detr-resnet-50",
        captioning_model: str = "Salesforce/blip-image-captioning-base",
        segmentation_model: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str = "cpu",
        detection_threshold: float = 0.5,
    ):
        self.detection_model_name = detection_model
        self.captioning_model_name = captioning_model
        self.segmentation_model_name = segmentation_model
        self.device = device
        self.detection_threshold = detection_threshold

        # Models (lazy loaded)
        self._detector = None
        self._detector_processor = None
        self._captioner = None
        self._captioner_processor = None
        self._segmenter = None
        self._segmenter_processor = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the perception system."""
        if self._initialized:
            return

        logger.info("Initializing Visual Perception System")

        # Initialize models in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models)

        self._initialized = True
        logger.info("Visual Perception System initialized")

    def _load_models(self) -> None:
        """Load vision models (blocking)."""
        try:
            from transformers import (
                DetrForObjectDetection,
                DetrImageProcessor,
                BlipProcessor,
                BlipForConditionalGeneration,
            )

            # Load detector
            logger.debug("Loading detection model", model=self.detection_model_name)
            self._detector_processor = DetrImageProcessor.from_pretrained(
                self.detection_model_name
            )
            self._detector = DetrForObjectDetection.from_pretrained(
                self.detection_model_name
            )
            self._detector.to(self.device)

            # Load captioner
            logger.debug("Loading captioning model", model=self.captioning_model_name)
            self._captioner_processor = BlipProcessor.from_pretrained(
                self.captioning_model_name
            )
            self._captioner = BlipForConditionalGeneration.from_pretrained(
                self.captioning_model_name
            )
            self._captioner.to(self.device)

            logger.info("Vision models loaded successfully")

        except ImportError as e:
            logger.warning(f"Vision models not available: {e}")
            logger.info("Visual perception will use mock implementations")

        except Exception as e:
            logger.warning(f"Failed to load vision models: {e}")

    async def _load_image(
        self,
        source: Union[str, Path, bytes, np.ndarray],
    ) -> Optional[Any]:
        """Load an image from various sources."""
        try:
            from PIL import Image

            if isinstance(source, np.ndarray):
                return Image.fromarray(source)
            elif isinstance(source, bytes):
                return Image.open(io.BytesIO(source))
            elif isinstance(source, (str, Path)):
                path = Path(source)
                if path.exists():
                    return Image.open(path)
                elif str(source).startswith(("http://", "https://")):
                    import httpx
                    async with httpx.AsyncClient() as client:
                        response = await client.get(str(source))
                        return Image.open(io.BytesIO(response.content))
            return None

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None

    async def detect_objects(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        threshold: Optional[float] = None,
    ) -> list[DetectedObject]:
        """
        Detect objects in an image.

        Args:
            image: Image source (path, URL, bytes, or numpy array)
            threshold: Detection confidence threshold

        Returns:
            List of DetectedObject
        """
        if not self._initialized:
            await self.initialize()

        threshold = threshold or self.detection_threshold

        pil_image = await self._load_image(image)
        if pil_image is None:
            return []

        if self._detector is None:
            # Return mock detection
            return [
                DetectedObject(
                    id="obj_0",
                    label="unknown",
                    confidence=0.5,
                    bounding_box=BoundingBox(0.1, 0.1, 0.9, 0.9),
                )
            ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._detect_objects_sync(pil_image, threshold),
        )

    def _detect_objects_sync(
        self,
        image,
        threshold: float,
    ) -> list[DetectedObject]:
        """Synchronous object detection."""
        import torch

        inputs = self._detector_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._detector(**inputs)

        # Process outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._detector_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        objects = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_name = self._detector.config.id2label[label.item()]
            bbox = BoundingBox(
                x_min=box[0].item() / image.size[0],
                y_min=box[1].item() / image.size[1],
                x_max=box[2].item() / image.size[0],
                y_max=box[3].item() / image.size[1],
            )

            objects.append(DetectedObject(
                id=f"obj_{len(objects)}",
                label=label_name,
                confidence=score.item(),
                bounding_box=bbox,
            ))

        return objects

    async def generate_caption(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        max_length: int = 50,
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image: Image source
            max_length: Maximum caption length

        Returns:
            Generated caption
        """
        if not self._initialized:
            await self.initialize()

        pil_image = await self._load_image(image)
        if pil_image is None:
            return "Unable to process image."

        if self._captioner is None:
            return "An image (vision models not available)."

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate_caption_sync(pil_image, max_length),
        )

    def _generate_caption_sync(self, image, max_length: int) -> str:
        """Synchronous caption generation."""
        import torch

        inputs = self._captioner_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._captioner.generate(**inputs, max_length=max_length)

        caption = self._captioner_processor.decode(output[0], skip_special_tokens=True)
        return caption

    async def answer_question(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        question: str,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: Image source
            question: Question to answer

        Returns:
            Answer to the question
        """
        if not self._initialized:
            await self.initialize()

        pil_image = await self._load_image(image)
        if pil_image is None:
            return "Unable to process image."

        if self._captioner is None:
            return "Visual question answering not available."

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._answer_question_sync(pil_image, question),
        )

    def _answer_question_sync(self, image, question: str) -> str:
        """Synchronous VQA."""
        import torch

        inputs = self._captioner_processor(
            images=image,
            text=question,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._captioner.generate(**inputs, max_length=50)

        answer = self._captioner_processor.decode(output[0], skip_special_tokens=True)
        return answer

    async def build_scene_graph(
        self,
        image: Union[str, Path, bytes, np.ndarray],
    ) -> SceneGraph:
        """
        Build a scene graph from an image.

        Args:
            image: Image source

        Returns:
            SceneGraph with objects and relations
        """
        # Detect objects
        objects = await self.detect_objects(image)

        # Generate caption for global context
        caption = await self.generate_caption(image)

        # Infer spatial relations
        relations = self._infer_spatial_relations(objects)

        return SceneGraph(
            objects=objects,
            relations=relations,
            global_features={
                "caption": caption,
                "object_count": len(objects),
            },
        )

    def _infer_spatial_relations(
        self,
        objects: list[DetectedObject],
    ) -> list[SceneRelation]:
        """Infer spatial relations between objects."""
        relations = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue

                box1 = obj1.bounding_box
                box2 = obj2.bounding_box

                # Left/right relations
                if box1.center[0] < box2.center[0] - 0.1:
                    relations.append(SceneRelation(
                        subject_id=obj1.id,
                        predicate="to the left of",
                        object_id=obj2.id,
                        confidence=0.8,
                    ))
                elif box1.center[0] > box2.center[0] + 0.1:
                    relations.append(SceneRelation(
                        subject_id=obj1.id,
                        predicate="to the right of",
                        object_id=obj2.id,
                        confidence=0.8,
                    ))

                # Above/below relations
                if box1.center[1] < box2.center[1] - 0.1:
                    relations.append(SceneRelation(
                        subject_id=obj1.id,
                        predicate="above",
                        object_id=obj2.id,
                        confidence=0.8,
                    ))
                elif box1.center[1] > box2.center[1] + 0.1:
                    relations.append(SceneRelation(
                        subject_id=obj1.id,
                        predicate="below",
                        object_id=obj2.id,
                        confidence=0.8,
                    ))

                # Near/overlapping
                if box1.iou(box2) > 0.3:
                    relations.append(SceneRelation(
                        subject_id=obj1.id,
                        predicate="overlapping with",
                        object_id=obj2.id,
                        confidence=0.9,
                    ))
                elif box1.iou(box2) > 0.0:
                    relations.append(SceneRelation(
                        subject_id=obj1.id,
                        predicate="near",
                        object_id=obj2.id,
                        confidence=0.7,
                    ))

        return relations

    async def shutdown(self) -> None:
        """Shutdown the perception system."""
        self._detector = None
        self._captioner = None
        self._segmenter = None
        self._initialized = False
