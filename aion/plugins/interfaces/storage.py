"""
AION Storage Plugin Interface

Interface for plugins that provide storage backends.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from aion.plugins.interfaces.base import BasePlugin
from aion.plugins.types import PluginManifest, PluginType, SemanticVersion


class StorageOperation(str, Enum):
    """Types of storage operations."""

    GET = "get"
    SET = "set"
    DELETE = "delete"
    EXISTS = "exists"
    LIST = "list"
    SCAN = "scan"
    BATCH_GET = "batch_get"
    BATCH_SET = "batch_set"
    BATCH_DELETE = "batch_delete"
    TRANSACTION = "transaction"


@dataclass
class StorageCapabilities:
    """Capabilities of a storage backend."""

    supports_ttl: bool = False
    supports_transactions: bool = False
    supports_batch: bool = True
    supports_scan: bool = True
    supports_prefix: bool = True
    supports_tags: bool = False
    supports_versioning: bool = False
    max_key_length: int = 1024
    max_value_size_bytes: int = 10 * 1024 * 1024  # 10MB
    max_batch_size: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supports_ttl": self.supports_ttl,
            "supports_transactions": self.supports_transactions,
            "supports_batch": self.supports_batch,
            "supports_scan": self.supports_scan,
            "supports_prefix": self.supports_prefix,
            "supports_tags": self.supports_tags,
            "supports_versioning": self.supports_versioning,
            "max_key_length": self.max_key_length,
            "max_value_size_bytes": self.max_value_size_bytes,
            "max_batch_size": self.max_batch_size,
        }


@dataclass
class StorageItem:
    """A stored item with metadata."""

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    version: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @property
    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class StorageStats:
    """Statistics for a storage backend."""

    total_keys: int = 0
    total_size_bytes: int = 0
    operations_count: Dict[StorageOperation, int] = field(default_factory=dict)
    average_latency_ms: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_keys": self.total_keys,
            "total_size_bytes": self.total_size_bytes,
            "operations_count": {op.value: count for op, count in self.operations_count.items()},
            "average_latency_ms": self.average_latency_ms,
            "error_count": self.error_count,
        }


class StoragePlugin(BasePlugin):
    """
    Interface for storage backend plugins.

    Storage plugins provide alternative persistence backends
    for AION's state management.

    Implement:
    - connect(): Establish connection
    - disconnect(): Close connection
    - get/set/delete: Basic CRUD operations
    - list_keys: Key enumeration
    """

    def __init__(self):
        super().__init__()
        self._connected = False
        self._stats = StorageStats()

    @property
    def is_connected(self) -> bool:
        """Check if storage is connected."""
        return self._connected

    @property
    def stats(self) -> StorageStats:
        """Get storage statistics."""
        return self._stats

    # === Required Methods ===

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to storage.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close storage connection."""
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value by key.

        Args:
            key: Storage key

        Returns:
            Value if found, None otherwise
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set value with optional TTL.

        Args:
            key: Storage key
            value: Value to store
            ttl: Time-to-live in seconds
            tags: Optional tags for the item
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete value by key.

        Args:
            key: Storage key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Storage key

        Returns:
            True if exists
        """
        pass

    @abstractmethod
    async def list_keys(
        self,
        prefix: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> List[str]:
        """
        List keys with optional prefix filter.

        Args:
            prefix: Key prefix to filter by
            limit: Maximum keys to return
            offset: Number of keys to skip

        Returns:
            List of keys
        """
        pass

    # === Optional Methods ===

    def get_capabilities(self) -> StorageCapabilities:
        """Get storage backend capabilities."""
        return StorageCapabilities()

    async def get_item(self, key: str) -> Optional[StorageItem]:
        """
        Get full item with metadata.

        Args:
            key: Storage key

        Returns:
            StorageItem if found
        """
        value = await self.get(key)
        if value is None:
            return None
        return StorageItem(key=key, value=value)

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values.

        Args:
            keys: List of keys

        Returns:
            Dict of key -> value for found keys
        """
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Set multiple values.

        Args:
            items: Dict of key -> value
            ttl: Time-to-live in seconds
        """
        for key, value in items.items():
            await self.set(key, value, ttl)

    async def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys deleted
        """
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count

    async def clear(self, prefix: str = "") -> int:
        """
        Clear all keys with optional prefix.

        Args:
            prefix: Key prefix to clear

        Returns:
            Number of keys deleted
        """
        keys = await self.list_keys(prefix=prefix, limit=10000)
        return await self.delete_many(keys)

    async def scan(
        self,
        prefix: str = "",
        batch_size: int = 100,
    ) -> AsyncIterator[StorageItem]:
        """
        Scan all items with prefix.

        Args:
            prefix: Key prefix
            batch_size: Items per batch

        Yields:
            StorageItem for each key
        """
        offset = 0
        while True:
            keys = await self.list_keys(prefix=prefix, limit=batch_size, offset=offset)
            if not keys:
                break

            for key in keys:
                item = await self.get_item(key)
                if item:
                    yield item

            offset += len(keys)
            if len(keys) < batch_size:
                break

    async def health_check(self) -> Dict[str, Any]:
        """Check storage health."""
        try:
            test_key = "__health_check__"
            await self.set(test_key, "ok", ttl=1)
            value = await self.get(test_key)
            await self.delete(test_key)
            return {
                "healthy": value == "ok",
                "connected": self._connected,
                "stats": self._stats.to_dict(),
            }
        except Exception as e:
            return {
                "healthy": False,
                "connected": self._connected,
                "error": str(e),
            }

    # === Transaction Support ===

    async def begin_transaction(self) -> str:
        """
        Begin a transaction.

        Returns:
            Transaction ID

        Raises:
            NotImplementedError: If transactions not supported
        """
        raise NotImplementedError("Transactions not supported")

    async def commit_transaction(self, transaction_id: str) -> None:
        """Commit a transaction."""
        raise NotImplementedError("Transactions not supported")

    async def rollback_transaction(self, transaction_id: str) -> None:
        """Rollback a transaction."""
        raise NotImplementedError("Transactions not supported")


# === Example Implementation ===


class InMemoryStoragePlugin(StoragePlugin):
    """In-memory storage plugin implementation."""

    def __init__(self):
        super().__init__()
        self._data: Dict[str, StorageItem] = {}

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id="inmemory-storage",
            name="In-Memory Storage",
            version=SemanticVersion(1, 0, 0),
            description="Simple in-memory storage backend",
            plugin_type=PluginType.STORAGE,
            entry_point="inmemory_storage:InMemoryStoragePlugin",
            tags=["storage", "memory", "simple"],
        )

    def get_capabilities(self) -> StorageCapabilities:
        return StorageCapabilities(
            supports_ttl=True,
            supports_batch=True,
            supports_scan=True,
            supports_prefix=True,
            supports_tags=True,
        )

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._data.clear()
        self._connected = False

    async def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if item is None:
            return None
        if item.is_expired:
            del self._data[key]
            return None
        return item.value

    async def get_item(self, key: str) -> Optional[StorageItem]:
        item = self._data.get(key)
        if item is None:
            return None
        if item.is_expired:
            del self._data[key]
            return None
        return item

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        from datetime import timedelta

        expires_at = None
        if ttl:
            expires_at = datetime.now() + timedelta(seconds=ttl)

        existing = self._data.get(key)
        version = (existing.version + 1) if existing else 1

        self._data[key] = StorageItem(
            key=key,
            value=value,
            created_at=existing.created_at if existing else datetime.now(),
            updated_at=datetime.now(),
            expires_at=expires_at,
            version=version,
            tags=tags or {},
        )

    async def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        item = self._data.get(key)
        if item is None:
            return False
        if item.is_expired:
            del self._data[key]
            return False
        return True

    async def list_keys(
        self,
        prefix: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> List[str]:
        # Clean expired items
        now = datetime.now()
        expired = [k for k, v in self._data.items() if v.expires_at and v.expires_at < now]
        for k in expired:
            del self._data[k]

        # Filter and paginate
        keys = sorted([k for k in self._data.keys() if k.startswith(prefix)])
        return keys[offset : offset + limit]

    async def initialize(self, kernel, config: Dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config
        await self.connect()
        self._initialized = True

    async def shutdown(self) -> None:
        await self.disconnect()
        self._initialized = False
