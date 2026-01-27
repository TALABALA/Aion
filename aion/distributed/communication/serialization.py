"""
AION Distributed Communication - Message Serialization

Production-grade serialization layer for distributed message passing.
Supports JSON (default) with optional msgpack fast path, integrity
verification via checksums, and transparent handling of complex Python
types (datetime, enum, set, bytes).
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Optional msgpack fast-path
# ---------------------------------------------------------------------------

try:
    import msgpack  # type: ignore[import-untyped]

    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False


# ---------------------------------------------------------------------------
# Custom JSON encoder / decoder helpers
# ---------------------------------------------------------------------------


class _DistributedEncoder(json.JSONEncoder):
    """JSON encoder that handles AION distributed types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return {"__datetime__": True, "isoformat": obj.isoformat()}
        if isinstance(obj, Enum):
            return {"__enum__": True, "type": type(obj).__name__, "value": obj.value}
        if isinstance(obj, set):
            return {"__set__": True, "items": sorted(obj)}
        if isinstance(obj, frozenset):
            return {"__frozenset__": True, "items": sorted(obj)}
        if isinstance(obj, bytes):
            return {"__bytes__": True, "b64": base64.b64encode(obj).decode("ascii")}
        if is_dataclass(obj) and not isinstance(obj, type):
            return {"__dataclass__": True, "type": type(obj).__name__, "data": asdict(obj)}
        return super().default(obj)


def _object_hook(dct: Dict[str, Any]) -> Any:
    """Decode custom-encoded objects back to Python types."""
    if "__datetime__" in dct:
        return datetime.fromisoformat(dct["isoformat"])
    if "__enum__" in dct:
        # Enum reconstruction is handled at a higher level; keep as dict
        return dct
    if "__set__" in dct:
        return set(dct["items"])
    if "__frozenset__" in dct:
        return frozenset(dct["items"])
    if "__bytes__" in dct:
        return base64.b64decode(dct["b64"])
    return dct


# ---------------------------------------------------------------------------
# Enum registry for deserialization
# ---------------------------------------------------------------------------

def _build_enum_registry() -> Dict[str, Type[Enum]]:
    """Build a name -> class mapping for known distributed enum types."""
    from aion.distributed.types import (
        ConsistencyLevel,
        ConflictResolution,
        NodeCapability,
        NodeRole,
        NodeStatus,
        RaftMessageType,
        ReplicationMode,
        ShardingStrategy,
        TaskPriority,
        TaskStatus,
        TaskType,
    )

    return {
        cls.__name__: cls
        for cls in (
            NodeRole, NodeStatus, NodeCapability,
            TaskStatus, TaskPriority, TaskType,
            ConsistencyLevel, ReplicationMode, ConflictResolution,
            ShardingStrategy, RaftMessageType,
        )
    }


# ---------------------------------------------------------------------------
# MessageSerializer
# ---------------------------------------------------------------------------


class MessageSerializer:
    """
    JSON-based serialization for all AION distributed message types.

    Features:
    - Transparent datetime / enum / set / bytes serialization
    - Optional msgpack fast-path (auto-detected)
    - SHA-256 checksum verification for integrity
    - Convenience helpers for DistributedTask, NodeInfo, RaftLogEntry
    """

    def __init__(self, *, use_msgpack: bool = False, verify_checksum: bool = True) -> None:
        self._use_msgpack = use_msgpack and _HAS_MSGPACK
        self._verify_checksum = verify_checksum
        self._enum_registry: Optional[Dict[str, Type[Enum]]] = None
        self._logger = structlog.get_logger("aion.distributed.serialization")

    # -- Enum registry (lazy) ------------------------------------------------

    @property
    def _enums(self) -> Dict[str, Type[Enum]]:
        if self._enum_registry is None:
            self._enum_registry = _build_enum_registry()
        return self._enum_registry

    # -- Public API ----------------------------------------------------------

    def serialize(self, obj: Any) -> bytes:
        """Serialize *obj* to bytes with an integrity checksum."""
        if self._use_msgpack:
            return self._serialize_msgpack(obj)
        return self._serialize_json(obj)

    def deserialize(self, data: bytes, cls: Optional[Type[T]] = None) -> Any:
        """Deserialize *data* back to a Python object.

        If *cls* is a dataclass, the result is reconstructed into that type.
        """
        if self._use_msgpack:
            return self._deserialize_msgpack(data, cls)
        return self._deserialize_json(data, cls)

    # -- Task helpers --------------------------------------------------------

    def serialize_task(self, task: Any) -> Dict[str, Any]:
        """Serialize a DistributedTask to a plain dict."""
        from aion.distributed.types import DistributedTask

        if not isinstance(task, DistributedTask):
            raise TypeError(f"Expected DistributedTask, got {type(task).__name__}")
        raw = asdict(task)
        return json.loads(json.dumps(raw, cls=_DistributedEncoder))

    def deserialize_task(self, data: Dict[str, Any]) -> Any:
        """Reconstruct a DistributedTask from a dict."""
        from aion.distributed.types import DistributedTask, TaskPriority, TaskStatus

        data = dict(data)
        # Restore enums
        if "priority" in data:
            data["priority"] = TaskPriority(data["priority"])
        if "status" in data:
            data["status"] = TaskStatus(data["status"])
        # Restore datetimes
        for dt_field in ("created_at", "started_at", "completed_at", "deadline"):
            val = data.get(dt_field)
            if isinstance(val, str):
                data[dt_field] = datetime.fromisoformat(val)
            elif isinstance(val, dict) and "__datetime__" in val:
                data[dt_field] = datetime.fromisoformat(val["isoformat"])
        # Restore sets
        for set_field in ("required_capabilities", "excluded_nodes"):
            val = data.get(set_field)
            if isinstance(val, list):
                data[set_field] = set(val)
            elif isinstance(val, dict) and "__set__" in val:
                data[set_field] = set(val["items"])
        return DistributedTask(**{
            k: v for k, v in data.items()
            if k in {f.name for f in fields(DistributedTask)}
        })

    # -- Node helpers --------------------------------------------------------

    def serialize_node(self, node: Any) -> Dict[str, Any]:
        """Serialize a NodeInfo to a plain dict."""
        from aion.distributed.types import NodeInfo

        if not isinstance(node, NodeInfo):
            raise TypeError(f"Expected NodeInfo, got {type(node).__name__}")
        raw = asdict(node)
        return json.loads(json.dumps(raw, cls=_DistributedEncoder))

    def deserialize_node(self, data: Dict[str, Any]) -> Any:
        """Reconstruct a NodeInfo from a dict."""
        from aion.distributed.types import NodeInfo, NodeRole, NodeStatus

        data = dict(data)
        # Restore enums
        if "role" in data:
            val = data["role"]
            if isinstance(val, str):
                data["role"] = NodeRole(val)
            elif isinstance(val, dict) and "__enum__" in val:
                data["role"] = NodeRole(val["value"])
        if "status" in data:
            val = data["status"]
            if isinstance(val, str):
                data["status"] = NodeStatus(val)
            elif isinstance(val, dict) and "__enum__" in val:
                data["status"] = NodeStatus(val["value"])
        # Restore datetimes
        for dt_field in ("started_at", "last_heartbeat", "joined_at"):
            val = data.get(dt_field)
            if isinstance(val, str):
                data[dt_field] = datetime.fromisoformat(val)
            elif isinstance(val, dict) and "__datetime__" in val:
                data[dt_field] = datetime.fromisoformat(val["isoformat"])
        # Restore sets
        for set_field in ("capabilities", "tags"):
            val = data.get(set_field)
            if isinstance(val, list):
                data[set_field] = set(val)
            elif isinstance(val, dict) and "__set__" in val:
                data[set_field] = set(val["items"])
        return NodeInfo(**{
            k: v for k, v in data.items()
            if k in {f.name for f in fields(NodeInfo)}
        })

    # -- Raft entry helpers --------------------------------------------------

    def serialize_raft_entry(self, entry: Any) -> Dict[str, Any]:
        """Serialize a RaftLogEntry to a plain dict."""
        from aion.distributed.types import RaftLogEntry

        if not isinstance(entry, RaftLogEntry):
            raise TypeError(f"Expected RaftLogEntry, got {type(entry).__name__}")
        raw = asdict(entry)
        return json.loads(json.dumps(raw, cls=_DistributedEncoder))

    def deserialize_raft_entry(self, data: Dict[str, Any]) -> Any:
        """Reconstruct a RaftLogEntry from a dict."""
        from aion.distributed.types import RaftLogEntry

        data = dict(data)
        # Restore datetime
        val = data.get("timestamp")
        if isinstance(val, str):
            data["timestamp"] = datetime.fromisoformat(val)
        elif isinstance(val, dict) and "__datetime__" in val:
            data["timestamp"] = datetime.fromisoformat(val["isoformat"])
        return RaftLogEntry(**{
            k: v for k, v in data.items()
            if k in {f.name for f in fields(RaftLogEntry)}
        })

    # -- Checksum ------------------------------------------------------------

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """Compute SHA-256 hex digest of *data*."""
        return hashlib.sha256(data).hexdigest()

    def verify_integrity(self, data: bytes, expected_checksum: str) -> bool:
        """Return True if *data* matches *expected_checksum*."""
        actual = self.compute_checksum(data)
        return actual == expected_checksum

    # -- JSON path -----------------------------------------------------------

    def _serialize_json(self, obj: Any) -> bytes:
        payload = json.dumps(obj, cls=_DistributedEncoder, sort_keys=True)
        checksum = self.compute_checksum(payload.encode("utf-8"))
        envelope = json.dumps(
            {"payload": json.loads(payload), "checksum": checksum},
            cls=_DistributedEncoder,
            sort_keys=True,
        )
        return envelope.encode("utf-8")

    def _deserialize_json(self, data: bytes, cls: Optional[Type[T]] = None) -> Any:
        envelope = json.loads(data.decode("utf-8"), object_hook=_object_hook)
        payload = envelope.get("payload", envelope)
        checksum = envelope.get("checksum")

        if self._verify_checksum and checksum is not None:
            payload_bytes = json.dumps(
                payload, cls=_DistributedEncoder, sort_keys=True,
            ).encode("utf-8")
            if not self.verify_integrity(payload_bytes, checksum):
                self._logger.error("checksum_mismatch", expected=checksum)
                raise ValueError("Message integrity check failed: checksum mismatch")

        if cls is not None and is_dataclass(cls):
            return self._reconstruct_dataclass(payload, cls)
        return payload

    # -- msgpack path --------------------------------------------------------

    def _serialize_msgpack(self, obj: Any) -> bytes:
        # Convert via JSON round-trip to normalise custom types
        json_str = json.dumps(obj, cls=_DistributedEncoder, sort_keys=True)
        normalised = json.loads(json_str)
        raw = msgpack.packb(normalised, use_bin_type=True)  # type: ignore[name-defined]
        checksum = self.compute_checksum(raw)
        envelope = msgpack.packb(  # type: ignore[name-defined]
            {"payload": normalised, "checksum": checksum},
            use_bin_type=True,
        )
        return envelope

    def _deserialize_msgpack(self, data: bytes, cls: Optional[Type[T]] = None) -> Any:
        envelope = msgpack.unpackb(data, raw=False)  # type: ignore[name-defined]
        payload = envelope.get("payload", envelope)
        checksum = envelope.get("checksum")

        if self._verify_checksum and checksum is not None:
            raw = msgpack.packb(payload, use_bin_type=True)  # type: ignore[name-defined]
            if not self.verify_integrity(raw, checksum):
                raise ValueError("Message integrity check failed: checksum mismatch")

        if cls is not None and is_dataclass(cls):
            return self._reconstruct_dataclass(payload, cls)
        return payload

    # -- Dataclass reconstruction --------------------------------------------

    def _reconstruct_dataclass(self, data: Any, cls: Type[T]) -> T:
        """Best-effort reconstruction of a dataclass from a dict."""
        if not isinstance(data, dict):
            raise TypeError(f"Cannot reconstruct {cls.__name__} from {type(data).__name__}")
        valid_fields = {f.name for f in fields(cls)}
        filtered = {}
        for k, v in data.items():
            if k not in valid_fields:
                continue
            # Restore wrapped enum values
            if isinstance(v, dict) and "__enum__" in v:
                enum_cls = self._enums.get(v["type"])
                if enum_cls is not None:
                    v = enum_cls(v["value"])
            # Restore wrapped datetimes
            if isinstance(v, dict) and "__datetime__" in v:
                v = datetime.fromisoformat(v["isoformat"])
            # Restore wrapped sets
            if isinstance(v, dict) and "__set__" in v:
                v = set(v["items"])
            filtered[k] = v
        return cls(**filtered)  # type: ignore[return-value]
