"""
AION Persistence Backends

Database backend implementations:
- SQLite (development/embedded)
- PostgreSQL (production)
- Redis (caching layer)
"""

from aion.persistence.backends.base import (
    BaseBackend,
    BackendCapabilities,
)

__all__ = [
    "BaseBackend",
    "BackendCapabilities",
]
