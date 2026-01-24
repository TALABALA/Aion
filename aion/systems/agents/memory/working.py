"""
Working Memory System

Short-term memory with attention mechanisms for active task processing.
Implements a cognitive architecture inspired by human working memory.
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Callable
import heapq

import structlog

logger = structlog.get_logger()


class SlotType(Enum):
    """Types of memory slots."""

    GOAL = "goal"  # Current goal/objective
    CONTEXT = "context"  # Task context
    INPUT = "input"  # Current input
    OUTPUT = "output"  # Generated output
    INTERMEDIATE = "intermediate"  # Intermediate results
    RETRIEVED = "retrieved"  # Retrieved from long-term memory
    ATTENTION = "attention"  # Items under attention


@dataclass
class AttentionWeight:
    """Attention weight for a memory item."""

    slot_id: str
    weight: float  # 0-1
    relevance: float  # Semantic relevance
    recency: float  # Time-based decay
    importance: float  # Task importance
    computed_at: datetime = field(default_factory=datetime.now)

    @property
    def combined_score(self) -> float:
        """Compute combined attention score."""
        return (
            self.weight * 0.4 +
            self.relevance * 0.3 +
            self.recency * 0.2 +
            self.importance * 0.1
        )


@dataclass
class MemorySlot:
    """A slot in working memory."""

    id: str
    slot_type: SlotType
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[timedelta] = None  # Time to live
    priority: float = 0.5
    tags: list[str] = field(default_factory=list)
    attention: Optional[AttentionWeight] = None
    locked: bool = False  # Prevent eviction

    @property
    def is_expired(self) -> bool:
        """Check if slot has expired."""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + self.ttl

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "slot_type": self.slot_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "ttl": self.ttl.total_seconds() if self.ttl else None,
            "priority": self.priority,
            "tags": self.tags,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorySlot":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            slot_type=SlotType(data["slot_type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            accessed_at=datetime.fromisoformat(data["accessed_at"]) if data.get("accessed_at") else datetime.now(),
            access_count=data.get("access_count", 0),
            ttl=timedelta(seconds=data["ttl"]) if data.get("ttl") else None,
            priority=data.get("priority", 0.5),
            tags=data.get("tags", []),
            locked=data.get("locked", False),
        )


class WorkingMemory:
    """
    Working memory system with attention mechanisms.

    Features:
    - Limited capacity (Miller's Law: 7±2 items)
    - Attention-based retrieval
    - Automatic decay and eviction
    - Priority-based slot management
    - Context switching
    - Chunking support
    """

    def __init__(
        self,
        capacity: int = 7,  # Miller's Law
        decay_rate: float = 0.1,  # Attention decay per second
        attention_threshold: float = 0.2,  # Minimum attention to retain
        context_id: Optional[str] = None,
    ):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.attention_threshold = attention_threshold
        self.context_id = context_id or f"ctx-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Storage
        self._slots: dict[str, MemorySlot] = {}
        self._attention_heap: list[tuple[float, str]] = []  # (neg_score, slot_id)

        # Indexes
        self._type_index: dict[SlotType, set[str]] = {t: set() for t in SlotType}
        self._tag_index: dict[str, set[str]] = {}

        # Focus
        self._focus_slot: Optional[str] = None
        self._focus_history: list[str] = []

        # Statistics
        self._total_stored = 0
        self._total_evicted = 0

        self._lock = asyncio.Lock()
        self._decay_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize working memory."""
        # Start decay task
        self._decay_task = asyncio.create_task(self._decay_loop())
        logger.info("working_memory_initialized", capacity=self.capacity, context=self.context_id)

    async def shutdown(self) -> None:
        """Shutdown working memory."""
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
        logger.info("working_memory_shutdown")

    async def store(
        self,
        content: Any,
        slot_type: SlotType = SlotType.INTERMEDIATE,
        slot_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        ttl: Optional[timedelta] = None,
        priority: float = 0.5,
        tags: Optional[list[str]] = None,
        lock: bool = False,
    ) -> MemorySlot:
        """Store content in working memory."""
        async with self._lock:
            # Generate ID if not provided
            if slot_id is None:
                slot_id = f"wm-{self._total_stored}-{datetime.now().strftime('%H%M%S%f')}"

            # Check if updating existing slot
            existing = self._slots.get(slot_id)
            if existing:
                existing.content = content
                existing.metadata.update(metadata or {})
                existing.touch()
                existing.priority = priority
                if tags:
                    existing.tags.extend(tags)
                return existing

            # Check capacity
            while len(self._slots) >= self.capacity:
                await self._evict_one()

            # Create slot
            slot = MemorySlot(
                id=slot_id,
                slot_type=slot_type,
                content=content,
                metadata=metadata or {},
                ttl=ttl,
                priority=priority,
                tags=tags or [],
                locked=lock,
            )

            # Compute initial attention
            slot.attention = AttentionWeight(
                slot_id=slot_id,
                weight=1.0,  # Full attention initially
                relevance=priority,
                recency=1.0,
                importance=priority,
            )

            # Store
            self._slots[slot_id] = slot
            self._type_index[slot_type].add(slot_id)
            for tag in slot.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(slot_id)

            # Add to attention heap
            heapq.heappush(self._attention_heap, (-slot.attention.combined_score, slot_id))

            self._total_stored += 1

            logger.debug("slot_stored", slot_id=slot_id, type=slot_type.value)

            return slot

    async def retrieve(self, slot_id: str) -> Optional[MemorySlot]:
        """Retrieve a slot by ID."""
        slot = self._slots.get(slot_id)
        if slot:
            slot.touch()
            # Boost attention on access
            if slot.attention:
                slot.attention.weight = min(1.0, slot.attention.weight + 0.2)
                slot.attention.recency = 1.0
                slot.attention.computed_at = datetime.now()
        return slot

    async def retrieve_by_type(self, slot_type: SlotType) -> list[MemorySlot]:
        """Retrieve all slots of a type."""
        slot_ids = self._type_index.get(slot_type, set())
        slots = []
        for slot_id in slot_ids:
            slot = await self.retrieve(slot_id)
            if slot:
                slots.append(slot)
        return slots

    async def retrieve_by_tags(self, tags: list[str]) -> list[MemorySlot]:
        """Retrieve slots matching tags."""
        if not tags:
            return []

        matching_ids = None
        for tag in tags:
            tag_slots = self._tag_index.get(tag, set())
            if matching_ids is None:
                matching_ids = tag_slots.copy()
            else:
                matching_ids &= tag_slots

        if not matching_ids:
            return []

        slots = []
        for slot_id in matching_ids:
            slot = await self.retrieve(slot_id)
            if slot:
                slots.append(slot)

        return slots

    async def focus(self, slot_id: str) -> Optional[MemorySlot]:
        """Set focus to a specific slot."""
        slot = self._slots.get(slot_id)
        if not slot:
            return None

        # Save previous focus
        if self._focus_slot and self._focus_slot != slot_id:
            self._focus_history.append(self._focus_slot)
            if len(self._focus_history) > 10:
                self._focus_history.pop(0)

        self._focus_slot = slot_id
        slot.touch()

        # Boost attention for focused slot
        if slot.attention:
            slot.attention.weight = 1.0
            slot.attention.recency = 1.0

        logger.debug("focus_changed", slot_id=slot_id)

        return slot

    async def get_focus(self) -> Optional[MemorySlot]:
        """Get currently focused slot."""
        if self._focus_slot:
            return await self.retrieve(self._focus_slot)
        return None

    async def unfocus(self) -> None:
        """Remove focus."""
        self._focus_slot = None

    async def get_attended(self, k: int = 5) -> list[MemorySlot]:
        """Get top-k slots by attention."""
        # Update attention scores
        await self._update_attention()

        # Get top slots
        scored_slots = []
        for slot in self._slots.values():
            if slot.attention:
                scored_slots.append((slot.attention.combined_score, slot))

        scored_slots.sort(key=lambda x: x[0], reverse=True)

        return [slot for _, slot in scored_slots[:k]]

    async def query(
        self,
        query_fn: Callable[[MemorySlot], bool],
    ) -> list[MemorySlot]:
        """Query slots with a filter function."""
        results = []
        for slot in self._slots.values():
            if query_fn(slot):
                slot.touch()
                results.append(slot)
        return results

    async def delete(self, slot_id: str) -> bool:
        """Delete a slot."""
        async with self._lock:
            slot = self._slots.pop(slot_id, None)
            if not slot:
                return False

            # Remove from indexes
            self._type_index[slot.slot_type].discard(slot_id)
            for tag in slot.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(slot_id)

            # Clear focus if needed
            if self._focus_slot == slot_id:
                self._focus_slot = None

            return True

    async def clear(self, slot_type: Optional[SlotType] = None) -> int:
        """Clear slots, optionally by type."""
        async with self._lock:
            if slot_type is None:
                count = len(self._slots)
                self._slots.clear()
                self._type_index = {t: set() for t in SlotType}
                self._tag_index.clear()
                self._focus_slot = None
                return count

            slot_ids = list(self._type_index.get(slot_type, set()))
            for slot_id in slot_ids:
                await self.delete(slot_id)
            return len(slot_ids)

    async def chunk(
        self,
        slot_ids: list[str],
        chunk_id: Optional[str] = None,
        chunk_type: SlotType = SlotType.INTERMEDIATE,
    ) -> Optional[MemorySlot]:
        """
        Chunk multiple slots into a single slot.

        This is a key working memory operation that combines related
        items to reduce cognitive load.
        """
        slots = []
        for slot_id in slot_ids:
            slot = self._slots.get(slot_id)
            if slot:
                slots.append(slot)

        if not slots:
            return None

        # Combine content
        chunked_content = {
            "items": [
                {"id": s.id, "content": s.content, "type": s.slot_type.value}
                for s in slots
            ],
            "count": len(slots),
        }

        # Combine metadata
        chunked_metadata = {
            "chunked_from": [s.id for s in slots],
            "chunk_time": datetime.now().isoformat(),
        }
        for slot in slots:
            for key, value in slot.metadata.items():
                if key not in chunked_metadata:
                    chunked_metadata[key] = value

        # Combine tags
        all_tags = set()
        for slot in slots:
            all_tags.update(slot.tags)

        # Compute priority as max of chunked slots
        max_priority = max(s.priority for s in slots)

        # Create chunk
        chunk = await self.store(
            content=chunked_content,
            slot_type=chunk_type,
            slot_id=chunk_id,
            metadata=chunked_metadata,
            priority=max_priority,
            tags=list(all_tags),
        )

        # Remove original slots
        for slot in slots:
            if not slot.locked:
                await self.delete(slot.id)

        logger.debug("slots_chunked", chunk_id=chunk.id, original_count=len(slots))

        return chunk

    async def _update_attention(self) -> None:
        """Update attention weights for all slots."""
        now = datetime.now()

        for slot in self._slots.values():
            if slot.attention:
                # Decay based on time since last access
                time_since_access = (now - slot.accessed_at).total_seconds()
                decay = math.exp(-self.decay_rate * time_since_access)

                slot.attention.recency = decay
                slot.attention.computed_at = now

                # Boost if focused
                if slot.id == self._focus_slot:
                    slot.attention.weight = 1.0

    async def _evict_one(self) -> None:
        """Evict the lowest attention slot."""
        if not self._slots:
            return

        # Update attention
        await self._update_attention()

        # Find lowest attention non-locked slot
        lowest_slot = None
        lowest_score = float("inf")

        for slot in self._slots.values():
            if slot.locked:
                continue
            if slot.is_expired:
                # Expired slots are immediately evictable
                lowest_slot = slot
                break

            if slot.attention:
                score = slot.attention.combined_score
                if score < lowest_score:
                    lowest_score = score
                    lowest_slot = slot

        if lowest_slot:
            await self.delete(lowest_slot.id)
            self._total_evicted += 1
            logger.debug("slot_evicted", slot_id=lowest_slot.id)

    async def _decay_loop(self) -> None:
        """Background task for attention decay."""
        while True:
            try:
                await asyncio.sleep(1.0)  # Update every second

                async with self._lock:
                    # Update attention
                    await self._update_attention()

                    # Remove expired slots
                    expired = [
                        slot_id for slot_id, slot in self._slots.items()
                        if slot.is_expired and not slot.locked
                    ]

                    for slot_id in expired:
                        await self.delete(slot_id)
                        self._total_evicted += 1

                    # Remove very low attention slots if over capacity
                    while len(self._slots) > self.capacity:
                        await self._evict_one()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("decay_loop_error", error=str(e))

    def get_context_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of working memory context."""
        return {
            "context_id": self.context_id,
            "focus": self._focus_slot,
            "slots": {
                slot_id: {
                    "type": slot.slot_type.value,
                    "priority": slot.priority,
                    "attention": slot.attention.combined_score if slot.attention else 0,
                    "content_preview": str(slot.content)[:100],
                }
                for slot_id, slot in self._slots.items()
            },
            "capacity_used": len(self._slots),
            "capacity_max": self.capacity,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get working memory statistics."""
        type_counts = {t.value: len(ids) for t, ids in self._type_index.items()}

        avg_attention = 0.0
        if self._slots:
            avg_attention = sum(
                s.attention.combined_score for s in self._slots.values() if s.attention
            ) / len(self._slots)

        return {
            "context_id": self.context_id,
            "capacity": self.capacity,
            "used": len(self._slots),
            "available": self.capacity - len(self._slots),
            "total_stored": self._total_stored,
            "total_evicted": self._total_evicted,
            "by_type": type_counts,
            "focus": self._focus_slot,
            "avg_attention": avg_attention,
            "locked_count": sum(1 for s in self._slots.values() if s.locked),
        }
