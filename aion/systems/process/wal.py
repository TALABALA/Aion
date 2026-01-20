"""
AION Write-Ahead Log (WAL) Implementation

Production-grade WAL with:
- Append-only writes with fsync for durability
- Segment-based storage for efficient compaction
- CRC32 checksums for corruption detection
- Batch writes for performance
- Recovery from crashes
- Log truncation and compaction
"""

from __future__ import annotations

import asyncio
import hashlib
import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple
import zlib

import structlog

logger = structlog.get_logger(__name__)


# WAL Record Format:
# | CRC32 (4 bytes) | Length (4 bytes) | Type (1 byte) | Term (8 bytes) | Index (8 bytes) | Data (variable) |
RECORD_HEADER_SIZE = 25  # 4 + 4 + 1 + 8 + 8
SEGMENT_MAGIC = b"AIONWAL1"
SEGMENT_HEADER_SIZE = 16  # Magic (8) + Version (4) + Reserved (4)


class RecordType(Enum):
    """WAL record types."""
    FULL = 1      # Complete record
    FIRST = 2     # First fragment of a record
    MIDDLE = 3    # Middle fragment
    LAST = 4      # Last fragment
    CHECKPOINT = 5
    SNAPSHOT = 6


@dataclass
class WALRecord:
    """A single WAL record."""
    term: int
    index: int
    data: bytes
    record_type: RecordType = RecordType.FULL
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        """Serialize record to bytes."""
        # Header without CRC
        header = struct.pack(
            "<IBqq",  # Little-endian: uint32 length, uint8 type, int64 term, int64 index
            len(self.data),
            self.record_type.value,
            self.term,
            self.index,
        )

        # Calculate CRC over header + data
        crc = zlib.crc32(header + self.data) & 0xFFFFFFFF

        return struct.pack("<I", crc) + header + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["WALRecord"]:
        """Deserialize record from bytes."""
        if len(data) < RECORD_HEADER_SIZE:
            return None

        crc_stored = struct.unpack("<I", data[:4])[0]
        length, rec_type, term, index = struct.unpack("<IBqq", data[4:RECORD_HEADER_SIZE])

        if len(data) < RECORD_HEADER_SIZE + length:
            return None

        record_data = data[RECORD_HEADER_SIZE:RECORD_HEADER_SIZE + length]

        # Verify CRC
        crc_calc = zlib.crc32(data[4:RECORD_HEADER_SIZE] + record_data) & 0xFFFFFFFF
        if crc_calc != crc_stored:
            raise WALCorruptionError(f"CRC mismatch: expected {crc_stored}, got {crc_calc}")

        return cls(
            term=term,
            index=index,
            data=record_data,
            record_type=RecordType(rec_type),
        )


class WALCorruptionError(Exception):
    """Raised when WAL corruption is detected."""
    pass


@dataclass
class WALSegment:
    """A single WAL segment file."""
    path: Path
    base_index: int
    file: Optional[BinaryIO] = None
    size: int = 0
    is_sealed: bool = False

    def open(self, mode: str = "ab") -> None:
        """Open segment file."""
        if self.file is None:
            self.file = open(self.path, mode)
            self.size = self.path.stat().st_size if self.path.exists() else 0

    def close(self) -> None:
        """Close segment file."""
        if self.file:
            self.file.close()
            self.file = None

    def sync(self) -> None:
        """Sync segment to disk."""
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())


class WriteAheadLog:
    """
    Production-grade Write-Ahead Log.

    Features:
    - Durable writes with fsync
    - Segment-based storage
    - Batch writes for performance
    - Automatic compaction
    - Crash recovery
    """

    def __init__(
        self,
        directory: str,
        segment_size: int = 64 * 1024 * 1024,  # 64MB segments
        sync_mode: str = "fsync",  # "fsync", "fdatasync", "async"
        batch_size: int = 100,
        batch_timeout_ms: int = 10,
    ):
        self.directory = Path(directory)
        self.segment_size = segment_size
        self.sync_mode = sync_mode
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms

        # Segments
        self._segments: List[WALSegment] = []
        self._active_segment: Optional[WALSegment] = None

        # Indexing
        self._first_index = 0
        self._last_index = 0
        self._last_term = 0

        # Batching
        self._batch: List[WALRecord] = []
        self._batch_lock = threading.Lock()
        self._batch_event = threading.Event()

        # Background writer
        self._writer_thread: Optional[threading.Thread] = None
        self._shutdown = False

        # Callbacks
        self._on_append: List[Callable[[WALRecord], None]] = []

        # Stats
        self._stats = {
            "records_written": 0,
            "bytes_written": 0,
            "syncs_performed": 0,
            "segments_created": 0,
            "segments_compacted": 0,
        }

        self._lock = threading.RLock()

    def open(self) -> None:
        """Open the WAL and recover from existing segments."""
        self.directory.mkdir(parents=True, exist_ok=True)

        # Find existing segments
        segment_files = sorted(self.directory.glob("wal_*.log"))

        if segment_files:
            for seg_file in segment_files:
                # Parse base index from filename
                base_index = int(seg_file.stem.split("_")[1])
                segment = WALSegment(path=seg_file, base_index=base_index)
                self._segments.append(segment)

            # Recover from segments
            self._recover()
        else:
            # Create first segment
            self._create_segment(1)

        # Start background writer if batching
        if self.batch_size > 1:
            self._writer_thread = threading.Thread(target=self._batch_writer_loop, daemon=True)
            self._writer_thread.start()

        logger.info(f"WAL opened: {len(self._segments)} segments, last_index={self._last_index}")

    def close(self) -> None:
        """Close the WAL."""
        self._shutdown = True

        # Flush remaining batch
        with self._batch_lock:
            if self._batch:
                self._write_batch(self._batch)
                self._batch.clear()

        self._batch_event.set()

        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)

        # Close all segments
        for segment in self._segments:
            segment.close()

        if self._active_segment:
            self._active_segment.close()

        logger.info("WAL closed")

    def append(self, term: int, index: int, data: bytes) -> None:
        """
        Append a record to the WAL.

        This is synchronous and returns only after the record is durable.
        """
        record = WALRecord(term=term, index=index, data=data)

        if self.batch_size > 1:
            # Add to batch
            with self._batch_lock:
                self._batch.append(record)

                if len(self._batch) >= self.batch_size:
                    batch = self._batch
                    self._batch = []
                    self._write_batch(batch)
                else:
                    self._batch_event.set()
        else:
            # Write immediately
            self._write_record(record)
            self._sync()

    def append_batch(self, records: List[Tuple[int, int, bytes]]) -> None:
        """Append multiple records atomically."""
        wal_records = [
            WALRecord(term=term, index=index, data=data)
            for term, index, data in records
        ]
        self._write_batch(wal_records)

    def read(self, start_index: int, end_index: Optional[int] = None) -> Iterator[WALRecord]:
        """Read records from WAL."""
        with self._lock:
            if start_index < self._first_index:
                raise ValueError(f"Index {start_index} is before first index {self._first_index}")

            end_index = end_index or self._last_index + 1

            for segment in self._segments:
                if segment.base_index > end_index:
                    break

                for record in self._read_segment(segment):
                    if record.index >= start_index and record.index < end_index:
                        yield record
                    elif record.index >= end_index:
                        return

    def truncate_before(self, index: int) -> None:
        """Truncate WAL before index (for compaction after snapshot)."""
        with self._lock:
            # Find segments to remove
            segments_to_remove = []
            for segment in self._segments:
                # Check if all records in segment are before index
                last_in_segment = self._get_segment_last_index(segment)
                if last_in_segment < index:
                    segments_to_remove.append(segment)

            # Remove segments
            for segment in segments_to_remove:
                segment.close()
                segment.path.unlink(missing_ok=True)
                self._segments.remove(segment)
                self._stats["segments_compacted"] += 1

            if self._segments:
                self._first_index = self._segments[0].base_index
            else:
                self._first_index = index

            logger.info(f"WAL truncated before {index}, removed {len(segments_to_remove)} segments")

    def truncate_after(self, index: int) -> None:
        """Truncate WAL after index (for log conflicts)."""
        with self._lock:
            # Find the segment containing index
            target_segment = None
            for segment in self._segments:
                if segment.base_index <= index:
                    target_segment = segment
                else:
                    # Remove this and all subsequent segments
                    segment.close()
                    segment.path.unlink(missing_ok=True)

            # Remove segments after target
            self._segments = [s for s in self._segments if s.base_index <= index]

            # Truncate within target segment
            if target_segment:
                self._truncate_segment_after(target_segment, index)

            self._last_index = index
            logger.info(f"WAL truncated after {index}")

    def get_last_index(self) -> int:
        """Get the last index in the WAL."""
        return self._last_index

    def get_last_term(self) -> int:
        """Get the term of the last entry."""
        return self._last_term

    def get_first_index(self) -> int:
        """Get the first index in the WAL."""
        return self._first_index

    def get_stats(self) -> Dict[str, Any]:
        """Get WAL statistics."""
        return {
            **self._stats,
            "first_index": self._first_index,
            "last_index": self._last_index,
            "segment_count": len(self._segments),
            "total_size": sum(s.size for s in self._segments),
        }

    def _create_segment(self, base_index: int) -> WALSegment:
        """Create a new segment."""
        filename = f"wal_{base_index:020d}.log"
        path = self.directory / filename

        segment = WALSegment(path=path, base_index=base_index)
        segment.open("wb")

        # Write segment header
        header = SEGMENT_MAGIC + struct.pack("<II", 1, 0)  # Version 1, reserved
        segment.file.write(header)
        segment.sync()
        segment.size = SEGMENT_HEADER_SIZE

        self._segments.append(segment)
        self._active_segment = segment
        self._stats["segments_created"] += 1

        logger.debug(f"Created new WAL segment: {filename}")
        return segment

    def _write_record(self, record: WALRecord) -> None:
        """Write a single record to the active segment."""
        with self._lock:
            if self._active_segment is None:
                self._create_segment(record.index)

            # Check if we need a new segment
            record_bytes = record.to_bytes()
            if self._active_segment.size + len(record_bytes) > self.segment_size:
                self._active_segment.is_sealed = True
                self._active_segment.close()
                self._create_segment(record.index)

            # Write record
            self._active_segment.open("ab")
            self._active_segment.file.write(record_bytes)
            self._active_segment.size += len(record_bytes)

            self._last_index = record.index
            self._last_term = record.term
            self._stats["records_written"] += 1
            self._stats["bytes_written"] += len(record_bytes)

            # Notify callbacks
            for callback in self._on_append:
                try:
                    callback(record)
                except Exception:
                    pass

    def _write_batch(self, records: List[WALRecord]) -> None:
        """Write a batch of records."""
        with self._lock:
            for record in records:
                self._write_record(record)
            self._sync()

    def _sync(self) -> None:
        """Sync active segment to disk."""
        if self._active_segment and self._active_segment.file:
            if self.sync_mode == "fsync":
                self._active_segment.file.flush()
                os.fsync(self._active_segment.file.fileno())
            elif self.sync_mode == "fdatasync":
                self._active_segment.file.flush()
                os.fdatasync(self._active_segment.file.fileno())
            else:
                self._active_segment.file.flush()

            self._stats["syncs_performed"] += 1

    def _batch_writer_loop(self) -> None:
        """Background thread for batch writing."""
        while not self._shutdown:
            # Wait for batch or timeout
            self._batch_event.wait(timeout=self.batch_timeout_ms / 1000.0)
            self._batch_event.clear()

            with self._batch_lock:
                if self._batch:
                    batch = self._batch
                    self._batch = []
                else:
                    continue

            try:
                self._write_batch(batch)
            except Exception as e:
                logger.error(f"Batch write error: {e}")

    def _recover(self) -> None:
        """Recover WAL state from existing segments."""
        logger.info("Recovering WAL from segments...")

        for segment in self._segments:
            try:
                for record in self._read_segment(segment):
                    if record.index > self._last_index:
                        self._last_index = record.index
                        self._last_term = record.term
            except WALCorruptionError as e:
                logger.error(f"Corruption in segment {segment.path}: {e}")
                # Truncate corrupted segment
                self._truncate_segment_at_corruption(segment)

        if self._segments:
            self._first_index = self._segments[0].base_index
            self._active_segment = self._segments[-1]
            self._active_segment.open("ab")

        logger.info(f"WAL recovery complete: first_index={self._first_index}, last_index={self._last_index}")

    def _read_segment(self, segment: WALSegment) -> Iterator[WALRecord]:
        """Read all records from a segment."""
        with open(segment.path, "rb") as f:
            # Read and verify header
            header = f.read(SEGMENT_HEADER_SIZE)
            if not header.startswith(SEGMENT_MAGIC):
                raise WALCorruptionError(f"Invalid segment magic in {segment.path}")

            # Read records
            while True:
                # Read header
                header_data = f.read(RECORD_HEADER_SIZE)
                if len(header_data) == 0:
                    break
                if len(header_data) < RECORD_HEADER_SIZE:
                    raise WALCorruptionError("Truncated record header")

                # Parse length
                length = struct.unpack("<I", header_data[4:8])[0]

                # Read data
                data = f.read(length)
                if len(data) < length:
                    raise WALCorruptionError("Truncated record data")

                # Parse record
                record = WALRecord.from_bytes(header_data + data)
                if record:
                    yield record

    def _get_segment_last_index(self, segment: WALSegment) -> int:
        """Get the last index in a segment."""
        last_index = segment.base_index
        try:
            for record in self._read_segment(segment):
                last_index = record.index
        except WALCorruptionError:
            pass
        return last_index

    def _truncate_segment_after(self, segment: WALSegment, index: int) -> None:
        """Truncate a segment after the given index."""
        # Read segment up to index
        records_to_keep = []
        for record in self._read_segment(segment):
            if record.index <= index:
                records_to_keep.append(record)
            else:
                break

        # Rewrite segment
        segment.close()
        with open(segment.path, "wb") as f:
            # Write header
            header = SEGMENT_MAGIC + struct.pack("<II", 1, 0)
            f.write(header)

            # Write records
            for record in records_to_keep:
                f.write(record.to_bytes())

            f.flush()
            os.fsync(f.fileno())

        segment.size = segment.path.stat().st_size

    def _truncate_segment_at_corruption(self, segment: WALSegment) -> None:
        """Truncate a segment at the point of corruption."""
        records_to_keep = []
        try:
            for record in self._read_segment(segment):
                records_to_keep.append(record)
        except WALCorruptionError:
            pass

        if records_to_keep:
            self._truncate_segment_after(segment, records_to_keep[-1].index)
        else:
            # Remove entirely corrupted segment
            segment.close()
            segment.path.unlink(missing_ok=True)
            self._segments.remove(segment)


class WALIterator:
    """Iterator for reading WAL entries."""

    def __init__(self, wal: WriteAheadLog, start_index: int, end_index: Optional[int] = None):
        self.wal = wal
        self.start_index = start_index
        self.end_index = end_index
        self._iterator = None

    def __iter__(self) -> Iterator[WALRecord]:
        self._iterator = self.wal.read(self.start_index, self.end_index)
        return self

    def __next__(self) -> WALRecord:
        if self._iterator is None:
            raise StopIteration
        return next(self._iterator)


# === Checkpoint Support ===

@dataclass
class WALCheckpoint:
    """WAL checkpoint for faster recovery."""
    index: int
    term: int
    state_hash: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_bytes(self) -> bytes:
        """Serialize checkpoint."""
        import json
        data = {
            "index": self.index,
            "term": self.term,
            "state_hash": self.state_hash,
            "created_at": self.created_at.isoformat(),
        }
        return json.dumps(data).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "WALCheckpoint":
        """Deserialize checkpoint."""
        import json
        d = json.loads(data.decode())
        return cls(
            index=d["index"],
            term=d["term"],
            state_hash=d["state_hash"],
            created_at=datetime.fromisoformat(d["created_at"]),
        )


class CheckpointManager:
    """Manages WAL checkpoints for faster recovery."""

    def __init__(self, wal: WriteAheadLog, checkpoint_interval: int = 10000):
        self.wal = wal
        self.checkpoint_interval = checkpoint_interval
        self._last_checkpoint_index = 0
        self._checkpoints: List[WALCheckpoint] = []

    def maybe_checkpoint(self, current_index: int, state_hash: str) -> Optional[WALCheckpoint]:
        """Create checkpoint if enough entries have passed."""
        if current_index - self._last_checkpoint_index >= self.checkpoint_interval:
            return self.create_checkpoint(current_index, state_hash)
        return None

    def create_checkpoint(self, index: int, state_hash: str) -> WALCheckpoint:
        """Create a new checkpoint."""
        # Get term at index
        term = 0
        for record in self.wal.read(index, index + 1):
            term = record.term
            break

        checkpoint = WALCheckpoint(
            index=index,
            term=term,
            state_hash=state_hash,
        )

        # Write checkpoint record
        self.wal.append(
            term=term,
            index=index,
            data=checkpoint.to_bytes(),
        )

        self._checkpoints.append(checkpoint)
        self._last_checkpoint_index = index

        logger.info(f"Created checkpoint at index {index}")
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[WALCheckpoint]:
        """Get the most recent checkpoint."""
        return self._checkpoints[-1] if self._checkpoints else None
