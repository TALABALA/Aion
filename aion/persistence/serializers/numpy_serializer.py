"""
AION NumPy Serializer

Efficient serialization of NumPy arrays with:
- Compression support (zlib, lz4, zstd)
- Dtype preservation
- Shape preservation
- Memory-efficient streaming
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np


class CompressionMethod(str, Enum):
    """Compression methods for numpy arrays."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class SerializedArray:
    """Metadata about a serialized array."""
    shape: tuple
    dtype: str
    compression: CompressionMethod
    original_size: int
    compressed_size: int


class NumpySerializer:
    """
    Efficient NumPy array serializer.

    Supports multiple compression methods and preserves
    all array metadata (shape, dtype).
    """

    def __init__(
        self,
        compression: CompressionMethod = CompressionMethod.ZLIB,
        compression_level: int = 6,
        compression_threshold: int = 1024,  # Only compress if larger
    ):
        self.compression = compression
        self.compression_level = compression_level
        self.compression_threshold = compression_threshold

    def serialize(
        self,
        array: np.ndarray,
        compress: Optional[bool] = None,
    ) -> tuple[bytes, SerializedArray]:
        """
        Serialize a NumPy array to bytes.

        Args:
            array: Array to serialize
            compress: Override compression (None = auto based on size)

        Returns:
            Tuple of (bytes, metadata)
        """
        # Ensure array is contiguous
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        # Serialize to bytes
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        data = buffer.getvalue()
        original_size = len(data)

        # Determine if we should compress
        should_compress = compress
        if should_compress is None:
            should_compress = original_size >= self.compression_threshold

        compression_used = CompressionMethod.NONE
        compressed_size = original_size

        if should_compress and self.compression != CompressionMethod.NONE:
            compressed_data = self._compress(data)
            if compressed_data and len(compressed_data) < original_size:
                data = compressed_data
                compression_used = self.compression
                compressed_size = len(data)

        metadata = SerializedArray(
            shape=array.shape,
            dtype=str(array.dtype),
            compression=compression_used,
            original_size=original_size,
            compressed_size=compressed_size,
        )

        return data, metadata

    def deserialize(
        self,
        data: bytes,
        metadata: Optional[SerializedArray] = None,
        compressed: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Deserialize bytes to a NumPy array.

        Args:
            data: Serialized bytes
            metadata: Optional metadata (if available)
            compressed: Whether data is compressed (None = auto-detect)

        Returns:
            NumPy array
        """
        # Determine if we need to decompress
        if compressed is None:
            if metadata:
                compressed = metadata.compression != CompressionMethod.NONE
            else:
                # Try to auto-detect by checking magic bytes
                compressed = not data.startswith(b'\x93NUMPY')

        if compressed:
            data = self._decompress(data, metadata.compression if metadata else None)

        # Deserialize
        buffer = io.BytesIO(data)
        return np.load(buffer, allow_pickle=False)

    def _compress(self, data: bytes) -> Optional[bytes]:
        """Compress data using configured method."""
        try:
            if self.compression == CompressionMethod.ZLIB:
                return zlib.compress(data, level=self.compression_level)

            elif self.compression == CompressionMethod.LZ4:
                try:
                    import lz4.frame
                    return lz4.frame.compress(data, compression_level=self.compression_level)
                except ImportError:
                    # Fall back to zlib
                    return zlib.compress(data, level=self.compression_level)

            elif self.compression == CompressionMethod.ZSTD:
                try:
                    import zstd
                    return zstd.compress(data, self.compression_level)
                except ImportError:
                    # Fall back to zlib
                    return zlib.compress(data, level=self.compression_level)

        except Exception:
            pass

        return None

    def _decompress(
        self,
        data: bytes,
        method: Optional[CompressionMethod] = None,
    ) -> bytes:
        """Decompress data."""
        if method == CompressionMethod.LZ4:
            try:
                import lz4.frame
                return lz4.frame.decompress(data)
            except ImportError:
                pass

        elif method == CompressionMethod.ZSTD:
            try:
                import zstd
                return zstd.decompress(data)
            except ImportError:
                pass

        # Default to zlib (most common)
        try:
            return zlib.decompress(data)
        except zlib.error:
            # Data might not be compressed
            return data

    def serialize_batch(
        self,
        arrays: list[np.ndarray],
    ) -> tuple[bytes, list[SerializedArray]]:
        """
        Serialize multiple arrays efficiently.

        Args:
            arrays: List of arrays to serialize

        Returns:
            Tuple of (combined bytes, list of metadata)
        """
        all_data = []
        all_metadata = []
        offsets = []
        current_offset = 0

        for array in arrays:
            data, metadata = self.serialize(array)
            offsets.append(current_offset)
            current_offset += len(data)
            all_data.append(data)
            all_metadata.append(metadata)

        combined = b''.join(all_data)
        return combined, all_metadata

    def deserialize_batch(
        self,
        data: bytes,
        metadata_list: list[SerializedArray],
    ) -> list[np.ndarray]:
        """
        Deserialize multiple arrays.

        Args:
            data: Combined bytes
            metadata_list: List of metadata for each array

        Returns:
            List of NumPy arrays
        """
        arrays = []
        offset = 0

        for metadata in metadata_list:
            chunk = data[offset:offset + metadata.compressed_size]
            array = self.deserialize(chunk, metadata)
            arrays.append(array)
            offset += metadata.compressed_size

        return arrays

    @staticmethod
    def estimate_size(
        shape: tuple,
        dtype: Union[str, np.dtype],
    ) -> int:
        """Estimate serialized size for an array shape and dtype."""
        dtype = np.dtype(dtype)
        element_count = np.prod(shape)
        element_size = dtype.itemsize
        # Add overhead for numpy header (~128 bytes)
        return int(element_count * element_size + 128)


# Convenience functions

def serialize_array(
    array: np.ndarray,
    compress: bool = True,
) -> bytes:
    """Serialize a numpy array to bytes."""
    serializer = NumpySerializer()
    data, _ = serializer.serialize(array, compress=compress)
    return data


def deserialize_array(
    data: bytes,
    compressed: bool = True,
) -> np.ndarray:
    """Deserialize bytes to a numpy array."""
    serializer = NumpySerializer()
    return serializer.deserialize(data, compressed=compressed)
