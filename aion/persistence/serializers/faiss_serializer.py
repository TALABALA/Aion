"""
AION FAISS Serializer

Serialization of FAISS indices with:
- Full index serialization
- ID mapping preservation
- Compression support
- Index metadata
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import json

import numpy as np


@dataclass
class SerializedIndex:
    """Metadata about a serialized FAISS index."""
    name: str
    index_type: str
    dimension: int
    num_vectors: int
    compressed: bool
    original_size: int
    compressed_size: int
    id_mapping_size: int


class FAISSSerializer:
    """
    FAISS index serializer with compression and metadata preservation.
    """

    def __init__(
        self,
        compression_level: int = 6,
    ):
        self.compression_level = compression_level

    def serialize(
        self,
        index,  # faiss.Index
        id_mapping: dict[int, str],
        name: str = "default",
        compress: bool = True,
    ) -> Tuple[bytes, bytes, SerializedIndex]:
        """
        Serialize a FAISS index and its ID mapping.

        Args:
            index: FAISS index to serialize
            id_mapping: Mapping from index positions to entity IDs
            name: Index name
            compress: Whether to compress the data

        Returns:
            Tuple of (index_bytes, mapping_bytes, metadata)
        """
        try:
            import faiss

            # Serialize FAISS index
            index_bytes = faiss.serialize_index(index).tobytes()
            original_size = len(index_bytes)

            # Determine index type
            index_type = self._get_index_type(index)

            # Compress if requested
            compressed = False
            compressed_size = original_size
            if compress:
                compressed_bytes = zlib.compress(index_bytes, level=self.compression_level)
                if len(compressed_bytes) < original_size:
                    index_bytes = compressed_bytes
                    compressed = True
                    compressed_size = len(index_bytes)

            # Serialize ID mapping
            mapping_json = json.dumps(id_mapping).encode('utf-8')
            if compress:
                mapping_json = zlib.compress(mapping_json, level=self.compression_level)

            metadata = SerializedIndex(
                name=name,
                index_type=index_type,
                dimension=index.d,
                num_vectors=index.ntotal,
                compressed=compressed,
                original_size=original_size,
                compressed_size=compressed_size,
                id_mapping_size=len(mapping_json),
            )

            return index_bytes, mapping_json, metadata

        except ImportError:
            raise ImportError("FAISS is required for index serialization")

    def deserialize(
        self,
        index_bytes: bytes,
        mapping_bytes: bytes,
        metadata: Optional[SerializedIndex] = None,
    ) -> Tuple[Any, dict[int, str]]:
        """
        Deserialize a FAISS index and its ID mapping.

        Args:
            index_bytes: Serialized index bytes
            mapping_bytes: Serialized ID mapping bytes
            metadata: Optional metadata

        Returns:
            Tuple of (index, id_mapping)
        """
        try:
            import faiss

            # Decompress if needed
            if metadata and metadata.compressed:
                index_bytes = zlib.decompress(index_bytes)
                mapping_bytes = zlib.decompress(mapping_bytes)
            else:
                # Try to decompress anyway (auto-detect)
                try:
                    index_bytes = zlib.decompress(index_bytes)
                except zlib.error:
                    pass
                try:
                    mapping_bytes = zlib.decompress(mapping_bytes)
                except zlib.error:
                    pass

            # Deserialize FAISS index
            index = faiss.deserialize_index(
                np.frombuffer(index_bytes, dtype=np.uint8)
            )

            # Deserialize ID mapping
            mapping_str = mapping_bytes.decode('utf-8')
            id_mapping = json.loads(mapping_str)

            # Convert string keys back to integers
            id_mapping = {int(k): v for k, v in id_mapping.items()}

            return index, id_mapping

        except ImportError:
            raise ImportError("FAISS is required for index deserialization")

    def _get_index_type(self, index) -> str:
        """Determine the type of FAISS index."""
        try:
            import faiss

            type_name = type(index).__name__

            if isinstance(index, faiss.IndexFlat):
                return "flat"
            elif isinstance(index, faiss.IndexIVFFlat):
                return "ivf_flat"
            elif isinstance(index, faiss.IndexIVFPQ):
                return "ivf_pq"
            elif isinstance(index, faiss.IndexHNSWFlat):
                return "hnsw"
            elif isinstance(index, faiss.IndexIDMap):
                return f"idmap_{self._get_index_type(index.index)}"
            else:
                return type_name.lower()

        except ImportError:
            return "unknown"

    @staticmethod
    def create_index(
        dimension: int,
        index_type: str = "flat",
        **kwargs,
    ):
        """
        Create a new FAISS index.

        Args:
            dimension: Vector dimension
            index_type: Type of index ("flat", "ivf", "hnsw")
            **kwargs: Additional arguments for specific index types

        Returns:
            FAISS index
        """
        try:
            import faiss

            if index_type == "flat":
                return faiss.IndexFlatL2(dimension)

            elif index_type == "ivf" or index_type == "ivf_flat":
                nlist = kwargs.get("nlist", 100)
                quantizer = faiss.IndexFlatL2(dimension)
                return faiss.IndexIVFFlat(quantizer, dimension, nlist)

            elif index_type == "hnsw":
                M = kwargs.get("M", 32)
                return faiss.IndexHNSWFlat(dimension, M)

            elif index_type == "ivf_pq":
                nlist = kwargs.get("nlist", 100)
                m = kwargs.get("m", 8)  # Number of subquantizers
                bits = kwargs.get("bits", 8)
                quantizer = faiss.IndexFlatL2(dimension)
                return faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)

            else:
                # Default to flat
                return faiss.IndexFlatL2(dimension)

        except ImportError:
            raise ImportError("FAISS is required for index creation")

    @staticmethod
    def estimate_index_size(
        num_vectors: int,
        dimension: int,
        index_type: str = "flat",
    ) -> int:
        """
        Estimate the memory size of an index.

        Args:
            num_vectors: Number of vectors
            dimension: Vector dimension
            index_type: Type of index

        Returns:
            Estimated size in bytes
        """
        # Size of vectors (float32)
        vector_size = num_vectors * dimension * 4

        # Add overhead based on index type
        if index_type == "flat":
            return vector_size + 1024  # Small overhead

        elif index_type in ("ivf", "ivf_flat"):
            # IVF has overhead for cluster centroids and inverted lists
            return int(vector_size * 1.1 + 100000)

        elif index_type == "hnsw":
            # HNSW has graph structure overhead
            return int(vector_size * 1.5)

        elif index_type == "ivf_pq":
            # PQ compresses vectors significantly
            return int(num_vectors * 32 + 100000)

        return vector_size


# Convenience functions

def serialize_faiss_index(
    index,
    id_mapping: dict[int, str],
    name: str = "default",
) -> Tuple[bytes, bytes, SerializedIndex]:
    """Serialize a FAISS index."""
    serializer = FAISSSerializer()
    return serializer.serialize(index, id_mapping, name)


def deserialize_faiss_index(
    index_bytes: bytes,
    mapping_bytes: bytes,
) -> Tuple[Any, dict[int, str]]:
    """Deserialize a FAISS index."""
    serializer = FAISSSerializer()
    return serializer.deserialize(index_bytes, mapping_bytes)
