"""
AION Migration Runner

State-of-the-art database migration system with:
- Version tracking in database
- Automatic discovery of migrations
- Up/down migration support
- Dry run mode
- Checksum validation
- Transaction safety
- Rollback capabilities
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class MigrationDirection(str, Enum):
    """Migration direction."""
    UP = "up"
    DOWN = "down"


@dataclass
class MigrationInfo:
    """Information about a migration."""
    version: str
    name: str
    description: str
    checksum: str
    applied_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None


@dataclass
class MigrationResult:
    """Result of running migrations."""
    success: bool
    applied: list[MigrationInfo] = field(default_factory=list)
    rolled_back: list[MigrationInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


class Migration(ABC):
    """Base class for migrations."""

    # Override these in subclasses
    version: str = "000"
    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    async def up(self, connection: Any) -> None:
        """Apply the migration."""
        pass

    @abstractmethod
    async def down(self, connection: Any) -> None:
        """Revert the migration."""
        pass

    def get_checksum(self) -> str:
        """Generate checksum of migration code."""
        import inspect
        source = inspect.getsource(self.__class__)
        return hashlib.sha256(source.encode()).hexdigest()[:16]


class DatabaseAdapter(Protocol):
    """Protocol for database connections."""

    async def execute(self, query: str, params: tuple = ()) -> None:
        """Execute a query."""
        ...

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Fetch all results."""
        ...

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        """Fetch one result."""
        ...


class MigrationRunner:
    """
    Executes database migrations with full version control.

    Features:
    - Automatic migration discovery
    - Version tracking in database
    - Transaction-safe execution
    - Dry run support
    - Rollback capabilities
    - Checksum validation
    """

    MIGRATIONS_TABLE = "aion_migrations"

    def __init__(
        self,
        migrations_dir: Optional[Path] = None,
        connection: Optional[Any] = None,
    ):
        self.migrations_dir = migrations_dir or (
            Path(__file__).parent / "versions"
        )
        self.connection = connection
        self._migrations: dict[str, Migration] = {}

    async def initialize(self, connection: Optional[Any] = None) -> None:
        """Initialize the migration system."""
        if connection:
            self.connection = connection

        if not self.connection:
            raise RuntimeError("No database connection provided")

        await self._create_migrations_table()
        self._discover_migrations()

    async def _create_migrations_table(self) -> None:
        """Create the migrations tracking table."""
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_time_ms REAL,
                rolled_back_at TIMESTAMP
            )
        """)

    def _discover_migrations(self) -> None:
        """Discover migration files in the migrations directory."""
        self._migrations.clear()

        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory does not exist: {self.migrations_dir}")
            return

        for path in sorted(self.migrations_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue

            try:
                migration = self._load_migration(path)
                if migration:
                    self._migrations[migration.version] = migration
                    logger.debug(f"Discovered migration: {migration.version} - {migration.name}")
            except Exception as e:
                logger.error(f"Failed to load migration {path}: {e}")

    def _load_migration(self, path: Path) -> Optional[Migration]:
        """Load a migration from a Python file."""
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Migration subclass in module
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type) and
                issubclass(obj, Migration) and
                obj is not Migration
            ):
                return obj()

        return None

    async def get_applied_migrations(self) -> list[MigrationInfo]:
        """Get list of applied migrations."""
        rows = await self.connection.fetch_all(f"""
            SELECT version, name, description, checksum, applied_at, execution_time_ms
            FROM {self.MIGRATIONS_TABLE}
            WHERE rolled_back_at IS NULL
            ORDER BY version ASC
        """)

        return [
            MigrationInfo(
                version=row["version"],
                name=row["name"],
                description=row["description"] or "",
                checksum=row["checksum"],
                applied_at=row["applied_at"],
                execution_time_ms=row["execution_time_ms"],
            )
            for row in rows
        ]

    async def get_pending_migrations(self) -> list[Migration]:
        """Get list of pending migrations."""
        applied = await self.get_applied_migrations()
        applied_versions = {m.version for m in applied}

        pending = [
            migration
            for version, migration in sorted(self._migrations.items())
            if version not in applied_versions
        ]

        return pending

    async def migrate(
        self,
        target_version: Optional[str] = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Run pending migrations up to target version.

        Args:
            target_version: Stop at this version (None = run all)
            dry_run: If True, don't actually apply migrations

        Returns:
            MigrationResult with details of what was applied
        """
        result = MigrationResult(success=True, dry_run=dry_run)

        pending = await self.get_pending_migrations()

        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        if not pending:
            logger.info("No pending migrations")
            return result

        logger.info(f"Found {len(pending)} pending migration(s)")

        for migration in pending:
            try:
                info = await self._apply_migration(migration, dry_run=dry_run)
                result.applied.append(info)
                logger.info(f"{'Would apply' if dry_run else 'Applied'} migration: {migration.version} - {migration.name}")
            except Exception as e:
                result.success = False
                result.errors.append(f"Migration {migration.version} failed: {e}")
                logger.error(f"Migration {migration.version} failed: {e}")
                break

        return result

    async def _apply_migration(
        self,
        migration: Migration,
        dry_run: bool = False,
    ) -> MigrationInfo:
        """Apply a single migration."""
        start_time = datetime.utcnow()

        if not dry_run:
            # Execute migration in transaction
            await self.connection.execute("BEGIN")
            try:
                await migration.up(self.connection)
                await self.connection.execute("COMMIT")
            except Exception:
                await self.connection.execute("ROLLBACK")
                raise

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        info = MigrationInfo(
            version=migration.version,
            name=migration.name,
            description=migration.description,
            checksum=migration.get_checksum(),
            applied_at=datetime.utcnow(),
            execution_time_ms=execution_time,
        )

        if not dry_run:
            await self.connection.execute(
                f"""
                INSERT INTO {self.MIGRATIONS_TABLE}
                (version, name, description, checksum, applied_at, execution_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    info.version,
                    info.name,
                    info.description,
                    info.checksum,
                    info.applied_at.isoformat(),
                    info.execution_time_ms,
                ),
            )

        return info

    async def rollback(
        self,
        steps: int = 1,
        target_version: Optional[str] = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Rollback migrations.

        Args:
            steps: Number of migrations to rollback
            target_version: Rollback to this version (exclusive)
            dry_run: If True, don't actually rollback

        Returns:
            MigrationResult with details of what was rolled back
        """
        result = MigrationResult(success=True, dry_run=dry_run)

        applied = await self.get_applied_migrations()
        applied.reverse()  # Most recent first

        if target_version:
            to_rollback = [m for m in applied if m.version > target_version]
        else:
            to_rollback = applied[:steps]

        if not to_rollback:
            logger.info("No migrations to rollback")
            return result

        logger.info(f"Rolling back {len(to_rollback)} migration(s)")

        for info in to_rollback:
            migration = self._migrations.get(info.version)
            if not migration:
                result.success = False
                result.errors.append(f"Migration {info.version} not found in migrations directory")
                break

            # Verify checksum
            if migration.get_checksum() != info.checksum:
                logger.warning(f"Checksum mismatch for migration {info.version}")

            try:
                await self._rollback_migration(migration, info, dry_run=dry_run)
                result.rolled_back.append(info)
                logger.info(f"{'Would rollback' if dry_run else 'Rolled back'} migration: {info.version} - {info.name}")
            except Exception as e:
                result.success = False
                result.errors.append(f"Rollback of {info.version} failed: {e}")
                logger.error(f"Rollback of {info.version} failed: {e}")
                break

        return result

    async def _rollback_migration(
        self,
        migration: Migration,
        info: MigrationInfo,
        dry_run: bool = False,
    ) -> None:
        """Rollback a single migration."""
        if not dry_run:
            await self.connection.execute("BEGIN")
            try:
                await migration.down(self.connection)

                # Mark as rolled back
                await self.connection.execute(
                    f"""
                    UPDATE {self.MIGRATIONS_TABLE}
                    SET rolled_back_at = ?
                    WHERE version = ?
                    """,
                    (datetime.utcnow().isoformat(), info.version),
                )

                await self.connection.execute("COMMIT")
            except Exception:
                await self.connection.execute("ROLLBACK")
                raise

    async def reset(self, dry_run: bool = False) -> MigrationResult:
        """
        Rollback all migrations.

        Args:
            dry_run: If True, don't actually rollback

        Returns:
            MigrationResult
        """
        applied = await self.get_applied_migrations()
        return await self.rollback(steps=len(applied), dry_run=dry_run)

    async def refresh(self, dry_run: bool = False) -> MigrationResult:
        """
        Reset and re-run all migrations.

        Args:
            dry_run: If True, don't actually execute

        Returns:
            Combined MigrationResult
        """
        reset_result = await self.reset(dry_run=dry_run)
        if not reset_result.success:
            return reset_result

        migrate_result = await self.migrate(dry_run=dry_run)

        # Combine results
        migrate_result.rolled_back = reset_result.rolled_back
        return migrate_result

    async def status(self) -> dict[str, Any]:
        """
        Get migration status.

        Returns:
            Dict with applied, pending, and current version
        """
        applied = await self.get_applied_migrations()
        pending = await self.get_pending_migrations()

        return {
            "current_version": applied[-1].version if applied else None,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied": [
                {
                    "version": m.version,
                    "name": m.name,
                    "applied_at": m.applied_at.isoformat() if m.applied_at else None,
                }
                for m in applied
            ],
            "pending": [
                {
                    "version": m.version,
                    "name": m.name,
                    "description": m.description,
                }
                for m in pending
            ],
        }

    async def validate(self) -> list[str]:
        """
        Validate migration integrity.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        applied = await self.get_applied_migrations()

        for info in applied:
            migration = self._migrations.get(info.version)

            if not migration:
                errors.append(f"Applied migration {info.version} not found in migrations directory")
                continue

            if migration.get_checksum() != info.checksum:
                errors.append(f"Checksum mismatch for migration {info.version}")

        # Check for version gaps
        versions = sorted([m.version for m in applied])
        for i, version in enumerate(versions[1:], 1):
            prev_version = versions[i - 1]
            # Simple gap detection - could be more sophisticated
            try:
                curr_num = int(version.split("_")[0])
                prev_num = int(prev_version.split("_")[0])
                if curr_num - prev_num > 1:
                    errors.append(f"Version gap detected between {prev_version} and {version}")
            except (ValueError, IndexError):
                pass

        return errors


# Convenience function for CLI usage
async def run_migrations(
    connection: Any,
    migrations_dir: Optional[Path] = None,
    target_version: Optional[str] = None,
    dry_run: bool = False,
) -> MigrationResult:
    """
    Convenience function to run migrations.

    Args:
        connection: Database connection
        migrations_dir: Path to migrations directory
        target_version: Optional target version
        dry_run: If True, don't apply migrations

    Returns:
        MigrationResult
    """
    runner = MigrationRunner(migrations_dir)
    await runner.initialize(connection)
    return await runner.migrate(target_version=target_version, dry_run=dry_run)
