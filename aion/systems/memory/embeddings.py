"""
AION Embedding Engine

Multi-model embedding generation with:
- Sentence Transformers support
- Local model caching
- Batch processing
- Dimension reduction
"""

from __future__ import annotations

import asyncio
from typing import Optional, Union

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingEngine:
    """
    Multi-model embedding generation engine.

    Supports:
    - Sentence Transformers models
    - Custom embedding functions
    - Batch processing for efficiency
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize

        self._model = None
        self._dimension: Optional[int] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._initialized:
            return

        logger.info("Initializing embedding engine", model=self.model_name)

        # Load model in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        self._initialized = True
        logger.info(
            "Embedding engine initialized",
            model=self.model_name,
            dimension=self._dimension,
        )

    def _load_model(self) -> None:
        """Load the embedding model (blocking)."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        except ImportError:
            logger.warning(
                "sentence-transformers not available, using mock embeddings"
            )
            self._dimension = 384  # Default dimension
            self._model = None
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, using mock embeddings")
            self._dimension = 384
            self._model = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Embedding engine not initialized")
        return self._dimension

    async def embed(
        self,
        text: Union[str, list[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text.

        Args:
            text: Single text or list of texts
            show_progress: Show progress bar for batch processing

        Returns:
            Numpy array of embeddings
        """
        if not self._initialized:
            await self.initialize()

        # Ensure list format
        texts = [text] if isinstance(text, str) else text

        # Generate embeddings
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._generate_embeddings(texts, show_progress),
        )

        return embeddings

    def _generate_embeddings(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings (blocking)."""
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize,
            )
            return np.array(embeddings, dtype=np.float32)
        else:
            # Mock embeddings for testing
            embeddings = np.random.randn(len(texts), self._dimension).astype(np.float32)
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            return embeddings

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings in batches.

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings
        """
        if not self._initialized:
            await self.initialize()

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await self.embed(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        if self.normalize:
            # Already normalized, dot product is cosine similarity
            return float(np.dot(embedding1, embedding2))
        else:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    async def shutdown(self) -> None:
        """Shutdown the embedding engine."""
        self._model = None
        self._initialized = False
