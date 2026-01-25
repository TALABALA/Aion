"""
AION State Manager

Central coordinator for all persistence operations:
- Repository lifecycle management
- Cross-repository transactions
- State snapshots and restore
- Health monitoring
- Metrics collection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from aion.persistence.config import PersistenceConfig, get_persistence_config
from aion.persistence.database import DatabaseManager
from aion.persistence.backends.redis_cache import CacheManager
from aion.persistence.repositories.memory_repo import MemoryRepository
from aion.persistence.repositories.planning_repo import PlanningRepository
from aion.persistence.repositories.process_repo import ProcessRepository, TaskRepository, EventRepository
from aion.persistence.repositories.evolution_repo import EvolutionRepository
from aion.persistence.repositories.tools_repo import ToolsRepository
from aion.persistence.repositories.config_repo import ConfigRepository

logger = structlog.get_logger(__name__)


@dataclass
class StateManagerStats:
    """Statistics for the state manager."""
    initialized_at: Optional[datetime] = None
    repositories_loaded: int = 0
    total_operations: int = 0
    failed_operations: int = 0
    last_backup_at: Optional[datetime] = None
    last_restore_at: Optional[datetime] = None
    cache_enabled: bool = False
    backup_enabled: bool = False


class StateManager:
    """
    Central state management for AION.

    Coordinates all persistence operations and provides
    a unified interface for state access across systems.

    Features:
    - Lazy repository initialization
    - Cross-repository transactions
    - State snapshots
    - Health monitoring
    - Automatic cleanup
    """

    def __init__(self, config: Optional[PersistenceConfig] = None):
        self.config = config or get_persistence_config()
        self._db: Optional[DatabaseManager] = None
        self._cache: Optional[CacheManager] = None

        # Repositories (lazy initialized)
        self._memories: Optional[MemoryRepository] = None
        self._plans: Optional[PlanningRepository] = None
        self._processes: Optional[ProcessRepository] = None
        self._tasks: Optional[TaskRepository] = None
        self._events: Optional[EventRepository] = None
        self._evolution: Optional[EvolutionRepository] = None
        self._tools: Optional[ToolsRepository] = None
        self._configs: Optional[ConfigRepository] = None

        # Backup manager
        self._backup: Optional["BackupManager"] = None

        # State
        self._initialized = False
        self._stats = StateManagerStats()
        self._lock = asyncio.Lock()

    # === Properties ===

    @property
    def db(self) -> DatabaseManager:
        """Get database manager."""
        if not self._db:
            raise RuntimeError("State manager not initialized")
        return self._db

    @property
    def cache(self) -> Optional[CacheManager]:
        """Get cache manager."""
        return self._cache

    @property
    def memories(self) -> MemoryRepository:
        """Get memory repository."""
        if not self._memories:
            raise RuntimeError("State manager not initialized")
        return self._memories

    @property
    def plans(self) -> PlanningRepository:
        """Get planning repository."""
        if not self._plans:
            raise RuntimeError("State manager not initialized")
        return self._plans

    @property
    def processes(self) -> ProcessRepository:
        """Get process repository."""
        if not self._processes:
            raise RuntimeError("State manager not initialized")
        return self._processes

    @property
    def tasks(self) -> TaskRepository:
        """Get task repository."""
        if not self._tasks:
            raise RuntimeError("State manager not initialized")
        return self._tasks

    @property
    def events(self) -> EventRepository:
        """Get event repository."""
        if not self._events:
            raise RuntimeError("State manager not initialized")
        return self._events

    @property
    def evolution(self) -> EvolutionRepository:
        """Get evolution repository."""
        if not self._evolution:
            raise RuntimeError("State manager not initialized")
        return self._evolution

    @property
    def tools(self) -> ToolsRepository:
        """Get tools repository."""
        if not self._tools:
            raise RuntimeError("State manager not initialized")
        return self._tools

    @property
    def configs(self) -> ConfigRepository:
        """Get config repository."""
        if not self._configs:
            raise RuntimeError("State manager not initialized")
        return self._configs

    @property
    def backup(self) -> Optional["BackupManager"]:
        """Get backup manager."""
        return self._backup

    @property
    def is_initialized(self) -> bool:
        """Check if state manager is initialized."""
        return self._initialized

    @property
    def stats(self) -> StateManagerStats:
        """Get state manager statistics."""
        return self._stats

    # === Lifecycle ===

    async def initialize(self) -> None:
        """Initialize the state manager and all repositories."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info(
                "Initializing State Manager",
                backend=self.config.backend.value,
            )

            # Validate configuration
            errors = self.config.validate()
            if errors:
                raise ValueError(f"Invalid persistence config: {errors}")

            # Initialize database
            self._db = DatabaseManager(self.config)
            await self._db.initialize()

            # Initialize cache if enabled
            if self.config.cache.enabled:
                self._cache = CacheManager(self.config.cache)
                await self._cache.initialize()
                self._stats.cache_enabled = True

            # Run migrations if enabled
            if self.config.migration.auto_migrate:
                from aion.persistence.migrations.runner import MigrationRunner
                runner = MigrationRunner(self._db)
                await runner.run_migrations()

            # Initialize repositories
            self._memories = MemoryRepository(self._db, self._cache)
            self._plans = PlanningRepository(self._db, self._cache)
            self._processes = ProcessRepository(self._db, self._cache)
            self._tasks = TaskRepository(self._db, self._cache)
            self._events = EventRepository(self._db)
            self._evolution = EvolutionRepository(self._db, self._cache)
            self._tools = ToolsRepository(self._db, self._cache)
            self._configs = ConfigRepository(self._db, self._cache)

            self._stats.repositories_loaded = 8

            # Initialize backup manager if enabled
            if self.config.backup.enabled:
                from aion.persistence.backup import BackupManager
                self._backup = BackupManager(self._db, self.config)
                self._stats.backup_enabled = True

            self._stats.initialized_at = datetime.now()
            self._initialized = True

            logger.info("State Manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the state manager."""
        if not self._initialized:
            return

        async with self._lock:
            logger.info("Shutting down State Manager")

            # Create final backup if configured
            if self._backup and self.config.backup.backup_on_shutdown:
                try:
                    await self._backup.create_backup("shutdown")
                    self._stats.last_backup_at = datetime.now()
                except Exception as e:
                    logger.error("Shutdown backup failed", error=str(e))

            # Shutdown cache
            if self._cache:
                await self._cache.shutdown()

            # Shutdown database
            if self._db:
                await self._db.shutdown()

            self._initialized = False
            logger.info("State Manager shutdown complete")

    # === Transactional Operations ===

    async def atomic(
        self,
        operations: list[tuple[str, Any, dict[str, Any]]],
    ) -> bool:
        """
        Execute multiple operations atomically.

        Args:
            operations: List of (repository_name, entity, kwargs) tuples

        Returns:
            True if all operations succeeded
        """
        async with self._db.transaction() as conn:
            try:
                for repo_name, entity, kwargs in operations:
                    repo = self._get_repository(repo_name)
                    operation = kwargs.pop("operation", "create")

                    if operation == "create":
                        await repo.create(entity)
                    elif operation == "update":
                        entity_id = kwargs.get("id", getattr(entity, "id", None))
                        await repo.update(entity_id, entity)
                    elif operation == "delete":
                        entity_id = kwargs.get("id", getattr(entity, "id", None))
                        await repo.delete(entity_id)

                self._stats.total_operations += len(operations)
                return True

            except Exception as e:
                self._stats.failed_operations += 1
                logger.error("Atomic operation failed", error=str(e))
                raise

    def _get_repository(self, name: str):
        """Get repository by name."""
        repos = {
            "memories": self._memories,
            "plans": self._plans,
            "processes": self._processes,
            "tasks": self._tasks,
            "evolution": self._evolution,
            "tools": self._tools,
            "configs": self._configs,
        }

        repo = repos.get(name)
        if not repo:
            raise ValueError(f"Unknown repository: {name}")
        return repo

    # === Snapshot Operations ===

    async def create_snapshot(self, name: str) -> str:
        """
        Create a full system state snapshot.

        Args:
            name: Snapshot name

        Returns:
            Snapshot ID
        """
        if not self._backup:
            raise RuntimeError("Backup manager not configured")

        snapshot_id = await self._backup.create_backup(name)
        self._stats.last_backup_at = datetime.now()
        return snapshot_id

    async def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore system state from a snapshot.

        Args:
            snapshot_id: Snapshot to restore

        Returns:
            True if restored successfully
        """
        if not self._backup:
            raise RuntimeError("Backup manager not configured")

        result = await self._backup.restore_backup(snapshot_id)
        if result:
            self._stats.last_restore_at = datetime.now()

            # Invalidate all caches
            if self._cache:
                await self._cache.invalidate_pattern("*")

        return result

    async def list_snapshots(self) -> list[dict[str, Any]]:
        """List available snapshots."""
        if not self._backup:
            return []
        return await self._backup.list_backups()

    # === Health & Stats ===

    async def health_check(self) -> dict[str, Any]:
        """Check persistence layer health."""
        health = {
            "healthy": True,
            "initialized": self._initialized,
            "timestamp": datetime.now().isoformat(),
        }

        if not self._initialized:
            health["healthy"] = False
            health["error"] = "State manager not initialized"
            return health

        # Database health
        try:
            db_health = await self._db.health_check()
            health["database"] = db_health
            if not db_health.get("healthy"):
                health["healthy"] = False
        except Exception as e:
            health["healthy"] = False
            health["database"] = {"healthy": False, "error": str(e)}

        # Cache health
        if self._cache:
            health["cache"] = {
                "enabled": True,
                "stats": self._cache.stats,
            }
        else:
            health["cache"] = {"enabled": False}

        # Repository counts
        try:
            health["repositories"] = {
                "memories": await self._memories.count() if self._memories else 0,
                "plans": await self._plans.count() if self._plans else 0,
                "processes": await self._processes.count() if self._processes else 0,
                "tasks": await self._tasks.count() if self._tasks else 0,
            }
        except Exception as e:
            logger.warning("Failed to get repository counts", error=str(e))

        return health

    def get_stats(self) -> dict[str, Any]:
        """Get state manager statistics."""
        stats = {
            "initialized_at": self._stats.initialized_at.isoformat() if self._stats.initialized_at else None,
            "repositories_loaded": self._stats.repositories_loaded,
            "total_operations": self._stats.total_operations,
            "failed_operations": self._stats.failed_operations,
            "last_backup_at": self._stats.last_backup_at.isoformat() if self._stats.last_backup_at else None,
            "last_restore_at": self._stats.last_restore_at.isoformat() if self._stats.last_restore_at else None,
            "cache_enabled": self._stats.cache_enabled,
            "backup_enabled": self._stats.backup_enabled,
        }

        if self._db:
            stats["database"] = self._db.stats

        if self._cache:
            stats["cache"] = {
                "hits": self._cache.stats.hits,
                "misses": self._cache.stats.misses,
                "hit_rate": self._cache.stats.hit_rate,
            }

        return stats

    # === Cleanup Operations ===

    async def cleanup(
        self,
        event_days: int = 30,
        execution_days: int = 30,
        snapshot_days: int = 7,
    ) -> dict[str, int]:
        """
        Clean up old data across all repositories.

        Args:
            event_days: Delete events older than this
            execution_days: Delete tool executions older than this
            snapshot_days: Delete performance snapshots older than this

        Returns:
            Counts of deleted items by type
        """
        results = {}

        # Clean events
        if self._events:
            results["events"] = await self._events.cleanup(older_than_days=event_days)

        # Clean tool executions
        if self._tools:
            results["tool_executions"] = await self._tools.cleanup_old_executions(days=execution_days)

        # Clean evolution data
        if self._evolution:
            evolution_results = await self._evolution.cleanup_old_data(
                snapshot_days=snapshot_days,
            )
            results.update(evolution_results)

        # Clean expired sessions
        if self._configs:
            results["sessions"] = await self._configs.cleanup_expired_sessions()

        logger.info("Cleanup completed", results=results)
        return results

    # === Convenience Methods ===

    async def get_system_state(self) -> dict[str, Any]:
        """Get a summary of the current system state."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "initialized": self._initialized,
        }

        if not self._initialized:
            return state

        # Get counts from each repository
        try:
            state["memories"] = {
                "count": await self._memories.count(),
                "stats": await self._memories.get_statistics(),
            }
        except Exception:
            pass

        try:
            state["plans"] = {
                "count": await self._plans.count(),
                "stats": await self._plans.get_global_stats(),
            }
        except Exception:
            pass

        try:
            state["processes"] = {
                "count": await self._processes.count(),
                "stats": await self._processes.get_statistics(),
            }
        except Exception:
            pass

        try:
            state["tools"] = {
                "stats": await self._tools.get_global_statistics(),
            }
        except Exception:
            pass

        try:
            state["evolution"] = {
                "stats": await self._evolution.get_statistics(),
            }
        except Exception:
            pass

        return state


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def set_state_manager(manager: StateManager) -> None:
    """Set the global state manager instance."""
    global _state_manager
    _state_manager = manager


async def init_state_manager(config: Optional[PersistenceConfig] = None) -> StateManager:
    """Initialize and return the global state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager(config)
    if not _state_manager.is_initialized:
        await _state_manager.initialize()
    return _state_manager
