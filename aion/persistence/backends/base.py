"""
AION Base Backend Interface

Abstract base classes for database backends defining
the common interface all backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BackendCapabilities:
    """Describes what a backend can do."""
    supports_json: bool = True
    supports_arrays: bool = True
    supports_transactions: bool = True
    supports_savepoints: bool = True
    supports_concurrent_writes: bool = True
    supports_full_text_search: bool = False
    supports_advisory_locks: bool = False
    supports_listen_notify: bool = False
    max_query_params: int = 999  # SQLite limit
    max_blob_size: int = 1_000_000_000  # 1GB


class BaseBackend(ABC):
    """
    Abstract base class for database backends.

    Defines the interface that all database backends must implement.
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Get backend capabilities."""
        pass

    @property
    @abstractmethod
    def placeholder(self) -> str:
        """Get parameter placeholder style (? or $1)."""
        pass

    @abstractmethod
    def convert_placeholders(self, query: str, count: int) -> str:
        """Convert placeholder style if needed."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the backend."""
        pass

    @abstractmethod
    async def execute_ddl(self, ddl: str) -> None:
        """Execute DDL statements (CREATE, ALTER, DROP)."""
        pass

    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        pass

    @abstractmethod
    async def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        """Get table column information."""
        pass

    @abstractmethod
    async def vacuum(self) -> None:
        """Optimize the database."""
        pass
