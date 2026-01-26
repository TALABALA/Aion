"""
AION Persistence Configuration

Comprehensive configuration for the persistence layer with:
- Multi-backend database settings
- Connection pooling configuration
- Encryption and compression options
- Event sourcing settings
- Backup and recovery options
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import os


class DatabaseBackend(str, Enum):
    """Supported database backends."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MEMORY = "memory"  # In-memory for testing


class CompressionType(str, Enum):
    """Compression algorithms for large blobs."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    ZSTD = "zstd"


class CacheStrategy(str, Enum):
    """Cache invalidation strategies."""
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"


@dataclass
class EncryptionConfig:
    """Configuration for encryption at rest."""
    enabled: bool = False
    algorithm: str = "AES-256-GCM"
    key_derivation: str = "PBKDF2"
    key_iterations: int = 100000
    key_path: Optional[Path] = None  # Path to encryption key file

    # Key rotation
    rotation_enabled: bool = False
    rotation_interval_days: int = 90


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration."""
    min_connections: int = 2
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_lifetime: float = 3600.0

    # Health checks
    health_check_interval: float = 30.0
    health_check_query: str = "SELECT 1"

    # Retry settings
    retry_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff_multiplier: float = 2.0


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for resilience."""
    enabled: bool = True
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class EventSourcingConfig:
    """Event sourcing configuration."""
    enabled: bool = True

    # Event store settings
    store_all_events: bool = True
    event_ttl_days: Optional[int] = 90  # None = forever

    # Snapshots
    snapshot_interval: int = 100  # Create snapshot every N events
    snapshot_retention: int = 10  # Keep last N snapshots

    # Change data capture
    cdc_enabled: bool = True
    cdc_batch_size: int = 100
    cdc_flush_interval: float = 5.0


@dataclass
class WALConfig:
    """Write-ahead log configuration."""
    enabled: bool = True
    sync_mode: str = "normal"  # "normal", "full", "off"
    checkpoint_interval: int = 1000  # Operations between checkpoints
    checkpoint_timeout: float = 30.0


@dataclass
class CacheConfig:
    """Cache layer configuration."""
    enabled: bool = True
    backend: str = "memory"  # "memory", "redis"
    strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH

    # Memory cache settings
    max_size_mb: int = 256
    ttl_seconds: int = 300

    # Redis settings (if backend is "redis")
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_prefix: str = "aion:"


@dataclass
class BackupConfig:
    """Backup configuration."""
    enabled: bool = True
    backup_dir: Path = field(default_factory=lambda: Path("./backups"))

    # Schedule
    auto_backup: bool = True
    backup_interval_hours: int = 24
    backup_on_shutdown: bool = True

    # Retention
    max_backups: int = 7
    compress_backups: bool = True

    # Cloud storage (optional)
    cloud_enabled: bool = False
    cloud_provider: Optional[str] = None  # "s3", "gcs", "azure"
    cloud_bucket: Optional[str] = None
    cloud_prefix: str = "aion-backups/"


@dataclass
class SQLiteConfig:
    """SQLite-specific configuration."""
    path: Path = field(default_factory=lambda: Path("./data/aion.db"))

    # Performance settings
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -64000  # 64MB (negative = KB)
    temp_store: str = "MEMORY"
    mmap_size: int = 268435456  # 256MB

    # Locking
    busy_timeout: int = 30000  # ms

    # Features
    foreign_keys: bool = True
    auto_vacuum: str = "INCREMENTAL"


@dataclass
class PostgreSQLConfig:
    """PostgreSQL-specific configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "aion"
    user: str = "aion"
    password: str = ""

    # Connection options
    ssl_mode: str = "prefer"  # "disable", "allow", "prefer", "require"
    ssl_cert: Optional[Path] = None
    ssl_key: Optional[Path] = None
    ssl_root_cert: Optional[Path] = None

    # Performance
    statement_cache_size: int = 100
    prepared_statement_cache_size: int = 256

    # Schema
    schema: str = "public"

    @property
    def connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        password_part = f":{self.password}" if self.password else ""
        return (
            f"postgresql://{self.user}{password_part}@"
            f"{self.host}:{self.port}/{self.database}"
        )


@dataclass
class MigrationConfig:
    """Migration configuration."""
    auto_migrate: bool = True
    migrations_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "migrations" / "versions"
    )

    # Safety
    backup_before_migrate: bool = True
    allow_downgrade: bool = False
    dry_run: bool = False


@dataclass
class PersistenceConfig:
    """
    Main persistence configuration.

    Combines all persistence-related settings into a single
    comprehensive configuration object.
    """
    # Backend selection
    backend: DatabaseBackend = DatabaseBackend.SQLITE

    # Backend-specific configs
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    postgresql: PostgreSQLConfig = field(default_factory=PostgreSQLConfig)

    # Connection management
    pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Data integrity
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    compression: CompressionType = CompressionType.ZLIB
    compression_threshold: int = 1024  # Compress blobs larger than this (bytes)

    # Durability
    wal: WALConfig = field(default_factory=WALConfig)
    event_sourcing: EventSourcingConfig = field(default_factory=EventSourcingConfig)

    # Performance
    cache: CacheConfig = field(default_factory=CacheConfig)
    batch_size: int = 100
    query_timeout: float = 30.0

    # Migrations
    migration: MigrationConfig = field(default_factory=MigrationConfig)

    # Backup
    backup: BackupConfig = field(default_factory=BackupConfig)

    # Logging
    log_queries: bool = False
    log_slow_queries: bool = True
    slow_query_threshold_ms: float = 100.0

    # Metrics
    enable_metrics: bool = True
    metrics_prefix: str = "aion_persistence"

    @classmethod
    def from_env(cls) -> "PersistenceConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Backend
        backend = os.environ.get("AION_DB_BACKEND", "sqlite").lower()
        config.backend = DatabaseBackend(backend)

        # SQLite
        if sqlite_path := os.environ.get("AION_SQLITE_PATH"):
            config.sqlite.path = Path(sqlite_path)

        # PostgreSQL
        if pg_host := os.environ.get("AION_PG_HOST"):
            config.postgresql.host = pg_host
        if pg_port := os.environ.get("AION_PG_PORT"):
            config.postgresql.port = int(pg_port)
        if pg_db := os.environ.get("AION_PG_DATABASE"):
            config.postgresql.database = pg_db
        if pg_user := os.environ.get("AION_PG_USER"):
            config.postgresql.user = pg_user
        if pg_pass := os.environ.get("AION_PG_PASSWORD"):
            config.postgresql.password = pg_pass

        # Pool
        if pool_min := os.environ.get("AION_POOL_MIN"):
            config.pool.min_connections = int(pool_min)
        if pool_max := os.environ.get("AION_POOL_MAX"):
            config.pool.max_connections = int(pool_max)

        # Cache
        if redis_host := os.environ.get("AION_REDIS_HOST"):
            config.cache.backend = "redis"
            config.cache.redis_host = redis_host
        if redis_port := os.environ.get("AION_REDIS_PORT"):
            config.cache.redis_port = int(redis_port)
        if redis_pass := os.environ.get("AION_REDIS_PASSWORD"):
            config.cache.redis_password = redis_pass

        # Encryption
        if os.environ.get("AION_ENCRYPTION_ENABLED", "").lower() == "true":
            config.encryption.enabled = True
        if key_path := os.environ.get("AION_ENCRYPTION_KEY_PATH"):
            config.encryption.key_path = Path(key_path)

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses

        def convert(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            return obj

        return convert(self)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate backend-specific settings
        if self.backend == DatabaseBackend.SQLITE:
            if not self.sqlite.path.parent.exists():
                try:
                    self.sqlite.path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create SQLite directory: {e}")

        elif self.backend == DatabaseBackend.POSTGRESQL:
            if not self.postgresql.host:
                errors.append("PostgreSQL host is required")
            if not self.postgresql.database:
                errors.append("PostgreSQL database name is required")

        # Validate pool settings
        if self.pool.min_connections > self.pool.max_connections:
            errors.append("Pool min_connections cannot exceed max_connections")

        if self.pool.min_connections < 1:
            errors.append("Pool min_connections must be at least 1")

        # Validate encryption
        if self.encryption.enabled and not self.encryption.key_path:
            errors.append("Encryption key path is required when encryption is enabled")

        # Validate backup
        if self.backup.enabled:
            try:
                self.backup.backup_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create backup directory: {e}")

        return errors


# Default configuration instance
_default_config: Optional[PersistenceConfig] = None


def get_persistence_config() -> PersistenceConfig:
    """Get the default persistence configuration."""
    global _default_config
    if _default_config is None:
        _default_config = PersistenceConfig.from_env()
    return _default_config


def set_persistence_config(config: PersistenceConfig) -> None:
    """Set the default persistence configuration."""
    global _default_config
    _default_config = config
