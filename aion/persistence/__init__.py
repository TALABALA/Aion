"""
AION Unified Persistence Layer

State-of-the-art persistence infrastructure providing:
- Multi-backend database support (SQLite, PostgreSQL, Redis)
- Event sourcing with change data capture
- Write-ahead logging for durability
- Optimistic locking for concurrency
- Circuit breaker pattern for resilience
- Connection pooling with health monitoring
- Compression for large blobs (embeddings)
- Encryption at rest support
- Automatic schema migrations
- Backup and restore capabilities
"""

from aion.persistence.config import (
    PersistenceConfig,
    DatabaseBackend,
    CompressionType,
    EncryptionConfig,
)
from aion.persistence.database import (
    DatabaseManager,
    DatabaseConnection,
    ConnectionPool,
)
from aion.persistence.state_manager import StateManager
from aion.persistence.transactions import (
    TransactionManager,
    Transaction,
    IsolationLevel,
)
from aion.persistence.backup import BackupManager
from aion.persistence.migrations.runner import MigrationRunner

# Repository exports
from aion.persistence.repositories.base import BaseRepository
from aion.persistence.repositories.memory_repo import MemoryRepository
from aion.persistence.repositories.planning_repo import PlanningRepository
from aion.persistence.repositories.process_repo import ProcessRepository, TaskRepository
from aion.persistence.repositories.evolution_repo import EvolutionRepository
from aion.persistence.repositories.tools_repo import ToolsRepository
from aion.persistence.repositories.config_repo import ConfigRepository

__all__ = [
    # Config
    "PersistenceConfig",
    "DatabaseBackend",
    "CompressionType",
    "EncryptionConfig",
    # Database
    "DatabaseManager",
    "DatabaseConnection",
    "ConnectionPool",
    # State Management
    "StateManager",
    # Transactions
    "TransactionManager",
    "Transaction",
    "IsolationLevel",
    # Backup
    "BackupManager",
    # Migrations
    "MigrationRunner",
    # Repositories
    "BaseRepository",
    "MemoryRepository",
    "PlanningRepository",
    "ProcessRepository",
    "TaskRepository",
    "EvolutionRepository",
    "ToolsRepository",
    "ConfigRepository",
]

__version__ = "1.0.0"
