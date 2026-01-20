"""
AION CRDT (Conflict-free Replicated Data Types) Implementation

State-based CRDTs (CvRDTs) for eventual consistency:
- G-Counter (grow-only counter)
- PN-Counter (positive-negative counter)
- LWW-Register (last-writer-wins register)
- MV-Register (multi-value register)
- G-Set (grow-only set)
- 2P-Set (two-phase set)
- OR-Set (observed-remove set)
- LWW-Map (last-writer-wins map)
- RGA (Replicated Growable Array) for collaborative text
"""

from __future__ import annotations

import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, Iterator, List, Optional, Set, Tuple, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CRDT(ABC):
    """Base class for all CRDTs."""

    @abstractmethod
    def merge(self, other: "CRDT") -> "CRDT":
        """Merge with another CRDT of the same type."""
        pass

    @abstractmethod
    def value(self) -> Any:
        """Get the current value."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CRDT":
        """Deserialize from dictionary."""
        pass


# === Counters ===

class GCounter(CRDT):
    """
    Grow-only Counter.

    Each node has its own counter that can only increase.
    The total value is the sum of all node counters.

    Supports: increment
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._counts: Dict[str, int] = {node_id: 0}

    def increment(self, amount: int = 1) -> None:
        """Increment the counter."""
        if amount < 0:
            raise ValueError("GCounter can only increment")
        self._counts[self.node_id] = self._counts.get(self.node_id, 0) + amount

    def value(self) -> int:
        """Get the current count."""
        return sum(self._counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        """Merge with another GCounter."""
        result = GCounter(self.node_id)
        all_nodes = set(self._counts.keys()) | set(other._counts.keys())

        for node in all_nodes:
            result._counts[node] = max(
                self._counts.get(node, 0),
                other._counts.get(node, 0),
            )

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "GCounter",
            "node_id": self.node_id,
            "counts": self._counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GCounter":
        counter = cls(data["node_id"])
        counter._counts = data["counts"]
        return counter


class PNCounter(CRDT):
    """
    Positive-Negative Counter.

    Two G-Counters: one for increments, one for decrements.
    The value is increments - decrements.

    Supports: increment, decrement
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._p = GCounter(node_id)  # Positive
        self._n = GCounter(node_id)  # Negative

    def increment(self, amount: int = 1) -> None:
        """Increment the counter."""
        if amount < 0:
            raise ValueError("Use decrement for negative values")
        self._p.increment(amount)

    def decrement(self, amount: int = 1) -> None:
        """Decrement the counter."""
        if amount < 0:
            raise ValueError("Use increment for negative values")
        self._n.increment(amount)

    def value(self) -> int:
        """Get the current count."""
        return self._p.value() - self._n.value()

    def merge(self, other: "PNCounter") -> "PNCounter":
        """Merge with another PNCounter."""
        result = PNCounter(self.node_id)
        result._p = self._p.merge(other._p)
        result._n = self._n.merge(other._n)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "PNCounter",
            "node_id": self.node_id,
            "p": self._p.to_dict(),
            "n": self._n.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PNCounter":
        counter = cls(data["node_id"])
        counter._p = GCounter.from_dict(data["p"])
        counter._n = GCounter.from_dict(data["n"])
        return counter


# === Registers ===

@dataclass
class Timestamp:
    """Logical timestamp for ordering."""
    time: float
    node_id: str
    counter: int = 0

    def __lt__(self, other: "Timestamp") -> bool:
        if self.time != other.time:
            return self.time < other.time
        if self.counter != other.counter:
            return self.counter < other.counter
        return self.node_id < other.node_id

    def __le__(self, other: "Timestamp") -> bool:
        return self < other or self == other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return False
        return (
            self.time == other.time and
            self.counter == other.counter and
            self.node_id == other.node_id
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "node_id": self.node_id,
            "counter": self.counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Timestamp":
        return cls(
            time=data["time"],
            node_id=data["node_id"],
            counter=data.get("counter", 0),
        )


class LWWRegister(CRDT, Generic[T]):
    """
    Last-Writer-Wins Register.

    Uses timestamps to determine which write wins.
    Simple but may lose concurrent updates.

    Supports: set, get
    """

    def __init__(self, node_id: str, value: Optional[T] = None):
        self.node_id = node_id
        self._value: Optional[T] = value
        self._timestamp = Timestamp(time=0, node_id=node_id)
        self._counter = 0

    def set(self, value: T) -> None:
        """Set the register value."""
        self._counter += 1
        self._value = value
        self._timestamp = Timestamp(
            time=time.time(),
            node_id=self.node_id,
            counter=self._counter,
        )

    def get(self) -> Optional[T]:
        """Get the register value."""
        return self._value

    def value(self) -> Optional[T]:
        """Get the register value."""
        return self._value

    def merge(self, other: "LWWRegister[T]") -> "LWWRegister[T]":
        """Merge with another LWWRegister."""
        result = LWWRegister(self.node_id)

        if other._timestamp > self._timestamp:
            result._value = other._value
            result._timestamp = other._timestamp
        else:
            result._value = self._value
            result._timestamp = self._timestamp

        result._counter = max(self._counter, other._counter)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LWWRegister",
            "node_id": self.node_id,
            "value": self._value,
            "timestamp": self._timestamp.to_dict(),
            "counter": self._counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWWRegister":
        register = cls(data["node_id"])
        register._value = data["value"]
        register._timestamp = Timestamp.from_dict(data["timestamp"])
        register._counter = data.get("counter", 0)
        return register


class MVRegister(CRDT, Generic[T]):
    """
    Multi-Value Register.

    Keeps all concurrent values; conflicts must be resolved by application.
    Better for preserving concurrent updates than LWW.

    Supports: set, get (returns set of concurrent values)
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._values: Dict[str, Tuple[T, "VectorClock"]] = {}
        self._clock = VectorClock(node_id)

    def set(self, value: T) -> None:
        """Set a value, replacing all current values."""
        self._clock.increment()

        # Create unique ID for this value
        value_id = f"{self.node_id}:{self._clock._clock[self.node_id]}"

        # Remove all dominated values
        self._values.clear()
        self._values[value_id] = (value, self._clock.copy())

    def get(self) -> Set[T]:
        """Get all concurrent values."""
        return {v for v, _ in self._values.values()}

    def value(self) -> Set[T]:
        """Get all concurrent values."""
        return self.get()

    def merge(self, other: "MVRegister[T]") -> "MVRegister[T]":
        """Merge with another MVRegister."""
        result = MVRegister(self.node_id)

        # Combine all values
        all_values = {**self._values, **other._values}

        # Keep only non-dominated values
        to_keep = {}
        for vid, (val, clock) in all_values.items():
            dominated = False
            for other_vid, (other_val, other_clock) in all_values.items():
                if vid != other_vid and clock < other_clock:
                    dominated = True
                    break

            if not dominated:
                to_keep[vid] = (val, clock)

        result._values = to_keep
        result._clock = self._clock.merge(other._clock)

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "MVRegister",
            "node_id": self.node_id,
            "values": {
                vid: {"value": val, "clock": clock.to_dict()}
                for vid, (val, clock) in self._values.items()
            },
            "clock": self._clock.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MVRegister":
        register = cls(data["node_id"])
        register._values = {
            vid: (d["value"], VectorClock.from_dict(d["clock"]))
            for vid, d in data["values"].items()
        }
        register._clock = VectorClock.from_dict(data["clock"])
        return register


# === Sets ===

class GSet(CRDT, Generic[T]):
    """
    Grow-only Set.

    Elements can only be added, never removed.

    Supports: add, contains, elements
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._elements: Set[T] = set()

    def add(self, element: T) -> None:
        """Add an element to the set."""
        self._elements.add(element)

    def contains(self, element: T) -> bool:
        """Check if element is in the set."""
        return element in self._elements

    def elements(self) -> Set[T]:
        """Get all elements."""
        return self._elements.copy()

    def value(self) -> Set[T]:
        """Get all elements."""
        return self.elements()

    def merge(self, other: "GSet[T]") -> "GSet[T]":
        """Merge with another GSet."""
        result = GSet(self.node_id)
        result._elements = self._elements | other._elements
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "GSet",
            "node_id": self.node_id,
            "elements": list(self._elements),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GSet":
        gset = cls(data["node_id"])
        gset._elements = set(data["elements"])
        return gset

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[T]:
        return iter(self._elements)


class TwoPSet(CRDT, Generic[T]):
    """
    Two-Phase Set.

    Elements can be added and removed, but once removed, cannot be re-added.

    Supports: add, remove, contains, elements
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._added: GSet[T] = GSet(node_id)
        self._removed: GSet[T] = GSet(node_id)

    def add(self, element: T) -> None:
        """Add an element to the set."""
        self._added.add(element)

    def remove(self, element: T) -> None:
        """Remove an element from the set (cannot be re-added)."""
        if element in self._added._elements:
            self._removed.add(element)

    def contains(self, element: T) -> bool:
        """Check if element is in the set."""
        return element in self._added._elements and element not in self._removed._elements

    def elements(self) -> Set[T]:
        """Get all elements."""
        return self._added._elements - self._removed._elements

    def value(self) -> Set[T]:
        """Get all elements."""
        return self.elements()

    def merge(self, other: "TwoPSet[T]") -> "TwoPSet[T]":
        """Merge with another 2P-Set."""
        result = TwoPSet(self.node_id)
        result._added = self._added.merge(other._added)
        result._removed = self._removed.merge(other._removed)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "TwoPSet",
            "node_id": self.node_id,
            "added": self._added.to_dict(),
            "removed": self._removed.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwoPSet":
        tpset = cls(data["node_id"])
        tpset._added = GSet.from_dict(data["added"])
        tpset._removed = GSet.from_dict(data["removed"])
        return tpset


class ORSet(CRDT, Generic[T]):
    """
    Observed-Remove Set.

    Elements can be added and removed freely.
    Uses unique tags to track add/remove operations.

    Supports: add, remove, contains, elements
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # Map from element to set of unique tags
        self._elements: Dict[T, Set[str]] = {}
        # Set of all removed tags
        self._tombstones: Set[str] = set()
        self._counter = 0

    def _generate_tag(self) -> str:
        """Generate a unique tag."""
        self._counter += 1
        return f"{self.node_id}:{self._counter}"

    def add(self, element: T) -> None:
        """Add an element to the set."""
        tag = self._generate_tag()
        if element not in self._elements:
            self._elements[element] = set()
        self._elements[element].add(tag)

    def remove(self, element: T) -> None:
        """Remove an element from the set."""
        if element in self._elements:
            # Add all current tags to tombstones
            self._tombstones.update(self._elements[element])
            # Clear the element's tags
            self._elements[element] = set()

    def contains(self, element: T) -> bool:
        """Check if element is in the set."""
        if element not in self._elements:
            return False
        # Element is present if it has any non-tombstoned tags
        return bool(self._elements[element] - self._tombstones)

    def elements(self) -> Set[T]:
        """Get all elements."""
        return {
            elem for elem, tags in self._elements.items()
            if tags - self._tombstones
        }

    def value(self) -> Set[T]:
        """Get all elements."""
        return self.elements()

    def merge(self, other: "ORSet[T]") -> "ORSet[T]":
        """Merge with another OR-Set."""
        result = ORSet(self.node_id)

        # Merge elements
        all_elements = set(self._elements.keys()) | set(other._elements.keys())
        for elem in all_elements:
            tags = self._elements.get(elem, set()) | other._elements.get(elem, set())
            if tags:
                result._elements[elem] = tags

        # Merge tombstones
        result._tombstones = self._tombstones | other._tombstones

        # Update counter
        result._counter = max(self._counter, other._counter)

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ORSet",
            "node_id": self.node_id,
            "elements": {str(k): list(v) for k, v in self._elements.items()},
            "tombstones": list(self._tombstones),
            "counter": self._counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ORSet":
        orset = cls(data["node_id"])
        orset._elements = {k: set(v) for k, v in data["elements"].items()}
        orset._tombstones = set(data["tombstones"])
        orset._counter = data["counter"]
        return orset

    def __len__(self) -> int:
        return len(self.elements())

    def __iter__(self) -> Iterator[T]:
        return iter(self.elements())


# === Maps ===

class LWWMap(CRDT, Generic[T]):
    """
    Last-Writer-Wins Map.

    A map where each key uses LWW semantics.

    Supports: set, get, delete, keys, values, items
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._entries: Dict[str, LWWRegister[T]] = {}
        self._tombstones: Dict[str, Timestamp] = {}

    def set(self, key: str, value: T) -> None:
        """Set a key-value pair."""
        if key not in self._entries:
            self._entries[key] = LWWRegister(self.node_id)
        self._entries[key].set(value)

        # Clear tombstone if present
        ts = self._entries[key]._timestamp
        if key in self._tombstones and ts > self._tombstones[key]:
            del self._tombstones[key]

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get a value by key."""
        if key not in self._entries:
            return default

        # Check if deleted
        if key in self._tombstones:
            entry_ts = self._entries[key]._timestamp
            if self._tombstones[key] > entry_ts:
                return default

        return self._entries[key].get()

    def delete(self, key: str) -> None:
        """Delete a key."""
        if key in self._entries:
            self._tombstones[key] = Timestamp(
                time=time.time(),
                node_id=self.node_id,
            )

    def keys(self) -> Set[str]:
        """Get all keys."""
        return {
            k for k in self._entries.keys()
            if k not in self._tombstones or
               self._entries[k]._timestamp > self._tombstones[k]
        }

    def values(self) -> List[T]:
        """Get all values."""
        return [self.get(k) for k in self.keys()]

    def items(self) -> List[Tuple[str, T]]:
        """Get all key-value pairs."""
        return [(k, self.get(k)) for k in self.keys()]

    def value(self) -> Dict[str, T]:
        """Get the map as a dict."""
        return {k: self.get(k) for k in self.keys()}

    def merge(self, other: "LWWMap[T]") -> "LWWMap[T]":
        """Merge with another LWWMap."""
        result = LWWMap(self.node_id)

        # Merge entries
        all_keys = set(self._entries.keys()) | set(other._entries.keys())
        for key in all_keys:
            if key in self._entries and key in other._entries:
                result._entries[key] = self._entries[key].merge(other._entries[key])
            elif key in self._entries:
                result._entries[key] = self._entries[key]
            else:
                result._entries[key] = other._entries[key]

        # Merge tombstones
        all_tombstones = set(self._tombstones.keys()) | set(other._tombstones.keys())
        for key in all_tombstones:
            self_ts = self._tombstones.get(key)
            other_ts = other._tombstones.get(key)

            if self_ts and other_ts:
                result._tombstones[key] = max(self_ts, other_ts)
            elif self_ts:
                result._tombstones[key] = self_ts
            else:
                result._tombstones[key] = other_ts

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LWWMap",
            "node_id": self.node_id,
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "tombstones": {k: v.to_dict() for k, v in self._tombstones.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWWMap":
        lwwmap = cls(data["node_id"])
        lwwmap._entries = {
            k: LWWRegister.from_dict(v)
            for k, v in data["entries"].items()
        }
        lwwmap._tombstones = {
            k: Timestamp.from_dict(v)
            for k, v in data["tombstones"].items()
        }
        return lwwmap


# === Vector Clock (used by CRDTs) ===

class VectorClock:
    """Vector clock for partial ordering."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._clock: Dict[str, int] = {node_id: 0}

    def increment(self) -> None:
        """Increment local clock."""
        self._clock[self.node_id] = self._clock.get(self.node_id, 0) + 1

    def copy(self) -> "VectorClock":
        """Create a copy."""
        vc = VectorClock(self.node_id)
        vc._clock = dict(self._clock)
        return vc

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge with another vector clock."""
        result = VectorClock(self.node_id)
        all_nodes = set(self._clock.keys()) | set(other._clock.keys())
        for node in all_nodes:
            result._clock[node] = max(
                self._clock.get(node, 0),
                other._clock.get(node, 0),
            )
        return result

    def __lt__(self, other: "VectorClock") -> bool:
        """Check if strictly happens-before."""
        less_or_equal = True
        strictly_less = False

        for node in set(self._clock.keys()) | set(other._clock.keys()):
            self_val = self._clock.get(node, 0)
            other_val = other._clock.get(node, 0)

            if self_val > other_val:
                less_or_equal = False
            if self_val < other_val:
                strictly_less = True

        return less_or_equal and strictly_less

    def __le__(self, other: "VectorClock") -> bool:
        for node in set(self._clock.keys()) | set(other._clock.keys()):
            if self._clock.get(node, 0) > other._clock.get(node, 0):
                return False
        return True

    def concurrent(self, other: "VectorClock") -> bool:
        """Check if concurrent (incomparable)."""
        return not (self <= other) and not (other <= self)

    def to_dict(self) -> Dict[str, int]:
        return dict(self._clock)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "VectorClock":
        node_id = next(iter(data.keys()), "unknown")
        vc = cls(node_id)
        vc._clock = dict(data)
        return vc


# === RGA for Collaborative Text ===

@dataclass
class RGANode:
    """A node in the RGA sequence."""
    id: str  # Unique identifier (timestamp.node_id)
    value: Optional[str]  # Character value (None if tombstone)
    left: Optional[str] = None  # ID of left neighbor when inserted
    timestamp: float = field(default_factory=time.time)

    def is_tombstone(self) -> bool:
        return self.value is None


class RGA(CRDT):
    """
    Replicated Growable Array.

    Used for collaborative text editing with proper interleaving
    of concurrent insertions.

    Supports: insert, delete, text, length
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # ID -> RGANode
        self._nodes: Dict[str, RGANode] = {}
        # Ordered list of node IDs
        self._order: List[str] = []
        self._counter = 0

    def _generate_id(self) -> str:
        """Generate unique ID."""
        self._counter += 1
        return f"{time.time()}.{self.node_id}.{self._counter}"

    def insert(self, index: int, char: str) -> str:
        """Insert a character at index. Returns the node ID."""
        # Find the left neighbor
        left_id = None
        if index > 0:
            visible_idx = 0
            for nid in self._order:
                if not self._nodes[nid].is_tombstone():
                    visible_idx += 1
                    if visible_idx == index:
                        left_id = nid
                        break

        node_id = self._generate_id()
        node = RGANode(
            id=node_id,
            value=char,
            left=left_id,
        )

        self._nodes[node_id] = node
        self._insert_ordered(node)

        return node_id

    def _insert_ordered(self, node: RGANode) -> None:
        """Insert node in correct position."""
        if node.left is None:
            # Insert at beginning
            insert_idx = 0
        else:
            # Find position after left neighbor
            left_idx = self._order.index(node.left)
            insert_idx = left_idx + 1

            # Skip any nodes that should come before this one
            while insert_idx < len(self._order):
                existing = self._nodes[self._order[insert_idx]]
                if existing.left == node.left:
                    # Same left neighbor - use timestamp to order
                    if existing.timestamp > node.timestamp:
                        break
                    elif existing.timestamp == node.timestamp:
                        # Use node ID as tiebreaker
                        if existing.id > node.id:
                            break
                else:
                    break
                insert_idx += 1

        self._order.insert(insert_idx, node.id)

    def delete(self, index: int) -> Optional[str]:
        """Delete character at index. Returns the deleted char."""
        visible_idx = 0
        for nid in self._order:
            node = self._nodes[nid]
            if not node.is_tombstone():
                if visible_idx == index:
                    char = node.value
                    node.value = None  # Tombstone
                    return char
                visible_idx += 1
        return None

    def text(self) -> str:
        """Get the current text."""
        return "".join(
            self._nodes[nid].value
            for nid in self._order
            if not self._nodes[nid].is_tombstone()
        )

    def value(self) -> str:
        """Get the current text."""
        return self.text()

    def length(self) -> int:
        """Get visible length."""
        return sum(
            1 for nid in self._order
            if not self._nodes[nid].is_tombstone()
        )

    def merge(self, other: "RGA") -> "RGA":
        """Merge with another RGA."""
        result = RGA(self.node_id)
        result._counter = max(self._counter, other._counter)

        # Combine all nodes
        for nid, node in self._nodes.items():
            result._nodes[nid] = RGANode(
                id=node.id,
                value=node.value,
                left=node.left,
                timestamp=node.timestamp,
            )

        for nid, node in other._nodes.items():
            if nid in result._nodes:
                # Keep tombstone if either is tombstoned
                if node.is_tombstone():
                    result._nodes[nid].value = None
            else:
                result._nodes[nid] = RGANode(
                    id=node.id,
                    value=node.value,
                    left=node.left,
                    timestamp=node.timestamp,
                )

        # Rebuild order
        result._order = []
        sorted_nodes = sorted(
            result._nodes.values(),
            key=lambda n: (n.timestamp, n.id),
        )

        for node in sorted_nodes:
            result._insert_ordered(node)

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "RGA",
            "node_id": self.node_id,
            "nodes": {
                nid: {
                    "id": n.id,
                    "value": n.value,
                    "left": n.left,
                    "timestamp": n.timestamp,
                }
                for nid, n in self._nodes.items()
            },
            "order": self._order,
            "counter": self._counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RGA":
        rga = cls(data["node_id"])
        rga._nodes = {
            nid: RGANode(
                id=n["id"],
                value=n["value"],
                left=n["left"],
                timestamp=n["timestamp"],
            )
            for nid, n in data["nodes"].items()
        }
        rga._order = data["order"]
        rga._counter = data["counter"]
        return rga


# === CRDT Manager ===

class CRDTManager:
    """
    Manages multiple CRDTs with automatic syncing.

    Provides:
    - Named CRDT instances
    - Periodic sync with peers
    - Delta-based synchronization
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._crdts: Dict[str, CRDT] = {}
        self._sync_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def create_counter(self, name: str) -> PNCounter:
        """Create a PN-Counter."""
        counter = PNCounter(self.node_id)
        self._crdts[name] = counter
        return counter

    def create_register(self, name: str) -> LWWRegister:
        """Create an LWW-Register."""
        register = LWWRegister(self.node_id)
        self._crdts[name] = register
        return register

    def create_set(self, name: str) -> ORSet:
        """Create an OR-Set."""
        orset = ORSet(self.node_id)
        self._crdts[name] = orset
        return orset

    def create_map(self, name: str) -> LWWMap:
        """Create an LWW-Map."""
        lwwmap = LWWMap(self.node_id)
        self._crdts[name] = lwwmap
        return lwwmap

    def create_text(self, name: str) -> RGA:
        """Create an RGA for text."""
        rga = RGA(self.node_id)
        self._crdts[name] = rga
        return rga

    def get(self, name: str) -> Optional[CRDT]:
        """Get a CRDT by name."""
        return self._crdts.get(name)

    def merge_remote(self, name: str, remote_state: Dict[str, Any]) -> None:
        """Merge remote state into local CRDT."""
        local = self._crdts.get(name)
        if not local:
            return

        # Reconstruct remote CRDT
        crdt_type = remote_state.get("type")
        if crdt_type == "GCounter":
            remote = GCounter.from_dict(remote_state)
        elif crdt_type == "PNCounter":
            remote = PNCounter.from_dict(remote_state)
        elif crdt_type == "LWWRegister":
            remote = LWWRegister.from_dict(remote_state)
        elif crdt_type == "MVRegister":
            remote = MVRegister.from_dict(remote_state)
        elif crdt_type == "GSet":
            remote = GSet.from_dict(remote_state)
        elif crdt_type == "TwoPSet":
            remote = TwoPSet.from_dict(remote_state)
        elif crdt_type == "ORSet":
            remote = ORSet.from_dict(remote_state)
        elif crdt_type == "LWWMap":
            remote = LWWMap.from_dict(remote_state)
        elif crdt_type == "RGA":
            remote = RGA.from_dict(remote_state)
        else:
            return

        # Merge
        self._crdts[name] = local.merge(remote)

    def get_state(self, name: str) -> Optional[Dict[str, Any]]:
        """Get serialized state of a CRDT."""
        crdt = self._crdts.get(name)
        if crdt:
            return crdt.to_dict()
        return None

    def on_sync(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Set callback for synchronization."""
        self._sync_callback = callback

    def sync_all(self) -> None:
        """Trigger sync of all CRDTs."""
        if not self._sync_callback:
            return

        for name, crdt in self._crdts.items():
            self._sync_callback(name, crdt.to_dict())
