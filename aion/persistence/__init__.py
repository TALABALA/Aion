"""
AION Unified Persistence Layer

True state-of-the-art persistence infrastructure providing:
- Multi-backend database support (SQLite, PostgreSQL, Redis)
- CQRS (Command Query Responsibility Segregation)
- Event sourcing with change data capture
- Vector database support (FAISS, pgvector)
- OpenTelemetry distributed tracing
- Prometheus metrics export
- Write-ahead logging for durability
- Optimistic locking for concurrency
- Circuit breaker pattern for resilience
- Connection pooling with health monitoring
- Compression for large blobs (embeddings)
- Encryption at rest support
- Automatic schema migrations
- Backup and restore with cloud support
- Outbox pattern for reliable messaging
- Query anomaly detection
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
    OptimisticLockError,
    PessimisticLockError,
    DeadlockError,
    UnitOfWork,
)
from aion.persistence.backup import BackupManager, BackupType, BackupStatus
from aion.persistence.migrations.runner import MigrationRunner
from aion.persistence.events import (
    EventStore,
    Event,
    CDCManager,
    CDCEvent,
    Aggregate,
    AggregateRepository,
)
from aion.persistence.cqrs import (
    CQRSCoordinator,
    Command,
    CommandResult,
    CommandBus,
    CommandHandler,
    Query,
    QueryResult,
    QueryBus,
    QueryHandler,
    ReadModel,
    ProjectionManager,
    OutboxProcessor,
    OutboxMessage,
)
from aion.persistence.observability import (
    ObservabilityCoordinator,
    TracingManager,
    MetricsCollector,
    QueryAnomalyDetector,
    HealthDashboard,
    DatabaseOperation,
)
from aion.persistence.vector_store import (
    VectorStore,
    VectorDocument,
    SearchResult,
    VectorIndexConfig,
    IndexType,
    DistanceMetric,
    FAISSVectorStore,
    VectorStoreFactory,
)

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
    "OptimisticLockError",
    "PessimisticLockError",
    "DeadlockError",
    "UnitOfWork",
    # Backup
    "BackupManager",
    "BackupType",
    "BackupStatus",
    # Migrations
    "MigrationRunner",
    # Event Sourcing
    "EventStore",
    "Event",
    "CDCManager",
    "CDCEvent",
    "Aggregate",
    "AggregateRepository",
    # CQRS
    "CQRSCoordinator",
    "Command",
    "CommandResult",
    "CommandBus",
    "CommandHandler",
    "Query",
    "QueryResult",
    "QueryBus",
    "QueryHandler",
    "ReadModel",
    "ProjectionManager",
    "OutboxProcessor",
    "OutboxMessage",
    # Observability
    "ObservabilityCoordinator",
    "TracingManager",
    "MetricsCollector",
    "QueryAnomalyDetector",
    "HealthDashboard",
    "DatabaseOperation",
    # Vector Store
    "VectorStore",
    "VectorDocument",
    "SearchResult",
    "VectorIndexConfig",
    "IndexType",
    "DistanceMetric",
    "FAISSVectorStore",
    "VectorStoreFactory",
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

__version__ = "2.0.0"
