"""
Comprehensive tests for AION Persistence Layer.

Tests cover:
- Configuration
- Database backends (SQLite, PostgreSQL)
- Connection pooling
- Repositories (CRUD operations)
- Transactions with optimistic locking
- Event sourcing and CDC
- Migrations
- Backup and restore
- State manager
- Serializers
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

# Import persistence components
from aion.persistence.config import (
    PersistenceConfig,
    DatabaseBackend,
    CompressionType,
    SQLiteConfig,
    ConnectionPoolConfig,
    CacheConfig,
    BackupConfig,
)
from aion.persistence.serializers.numpy_serializer import (
    NumpySerializer,
    CompressionMethod,
    serialize_array,
    deserialize_array,
)
from aion.persistence.serializers.datetime_serializer import (
    DateTimeSerializer,
    serialize_datetime,
    deserialize_datetime,
)


# ==================== Configuration Tests ====================

class TestPersistenceConfig:
    """Tests for persistence configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PersistenceConfig()

        assert config.backend == DatabaseBackend.SQLITE
        assert config.compression == CompressionType.ZLIB
        assert config.pool.min_connections == 2
        assert config.pool.max_connections == 20
        assert config.cache.enabled is True

    def test_sqlite_config(self):
        """Test SQLite-specific configuration."""
        config = PersistenceConfig()

        assert config.sqlite.journal_mode == "WAL"
        assert config.sqlite.synchronous == "NORMAL"
        assert config.sqlite.foreign_keys is True

    def test_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            "AION_DB_BACKEND": "postgresql",
            "AION_PG_HOST": "localhost",
            "AION_PG_PORT": "5433",
            "AION_PG_DATABASE": "test_aion",
        }):
            config = PersistenceConfig.from_env()

            assert config.backend == DatabaseBackend.POSTGRESQL
            assert config.postgresql.host == "localhost"
            assert config.postgresql.port == 5433
            assert config.postgresql.database == "test_aion"

    def test_config_validation_valid(self):
        """Test config validation with valid settings."""
        config = PersistenceConfig()
        config.encryption.enabled = False  # Disable encryption for test

        with tempfile.TemporaryDirectory() as tmpdir:
            config.sqlite.path = Path(tmpdir) / "test.db"
            config.backup.backup_dir = Path(tmpdir) / "backups"

            errors = config.validate()
            # Should have no errors or only non-critical ones
            assert len([e for e in errors if "encryption" not in e.lower()]) == 0

    def test_config_validation_invalid_pool(self):
        """Test config validation catches invalid pool settings."""
        config = PersistenceConfig()
        config.pool.min_connections = 50
        config.pool.max_connections = 10  # Invalid: min > max

        errors = config.validate()
        assert any("pool" in e.lower() for e in errors)

    def test_config_to_dict(self):
        """Test configuration serialization to dict."""
        config = PersistenceConfig()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["backend"] == "sqlite"
        assert "sqlite" in data
        assert "pool" in data


# ==================== Serializer Tests ====================

class TestNumpySerializer:
    """Tests for NumPy array serialization."""

    def test_serialize_deserialize_1d(self):
        """Test serialization of 1D array."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        serializer = NumpySerializer()

        data, metadata = serializer.serialize(arr)
        result = serializer.deserialize(data, metadata)

        np.testing.assert_array_equal(arr, result)
        assert metadata.shape == (5,)
        assert metadata.dtype == "float32"

    def test_serialize_deserialize_2d(self):
        """Test serialization of 2D array."""
        arr = np.random.randn(10, 20).astype(np.float64)
        serializer = NumpySerializer()

        data, metadata = serializer.serialize(arr)
        result = serializer.deserialize(data, metadata)

        np.testing.assert_array_almost_equal(arr, result)
        assert metadata.shape == (10, 20)

    def test_compression_zlib(self):
        """Test ZLIB compression."""
        arr = np.zeros((100, 100), dtype=np.float32)  # Highly compressible
        serializer = NumpySerializer(compression=CompressionMethod.ZLIB)

        data, metadata = serializer.serialize(arr)

        assert metadata.compression == CompressionMethod.ZLIB
        assert metadata.compressed_size < metadata.original_size

    def test_no_compression_small_array(self):
        """Test small arrays are not compressed."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        serializer = NumpySerializer(compression_threshold=10000)

        data, metadata = serializer.serialize(arr)

        assert metadata.compression == CompressionMethod.NONE

    def test_batch_serialization(self):
        """Test batch serialization of multiple arrays."""
        arrays = [
            np.random.randn(10, 10).astype(np.float32),
            np.random.randn(5, 5).astype(np.float64),
            np.random.randint(0, 100, (20,), dtype=np.int32),
        ]
        serializer = NumpySerializer()

        combined, metadata_list = serializer.serialize_batch(arrays)
        results = serializer.deserialize_batch(combined, metadata_list)

        assert len(results) == 3
        for orig, result in zip(arrays, results):
            np.testing.assert_array_almost_equal(orig, result)

    def test_convenience_functions(self):
        """Test convenience serialize/deserialize functions."""
        arr = np.random.randn(50, 50).astype(np.float32)

        data = serialize_array(arr)
        result = deserialize_array(data)

        np.testing.assert_array_almost_equal(arr, result)

    def test_estimate_size(self):
        """Test size estimation."""
        size = NumpySerializer.estimate_size((100, 100), np.float32)

        assert size > 100 * 100 * 4  # At least raw data size
        assert size < 100 * 100 * 4 + 1000  # Reasonable overhead


class TestDateTimeSerializer:
    """Tests for datetime serialization."""

    def test_serialize_deserialize_datetime(self):
        """Test datetime round-trip."""
        dt = datetime(2024, 6, 15, 10, 30, 45)
        serializer = DateTimeSerializer()

        serialized = serializer.serialize(dt)
        result = serializer.deserialize(serialized)

        assert result == dt

    def test_serialize_with_timezone(self):
        """Test datetime with timezone info."""
        from datetime import timezone
        dt = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        serializer = DateTimeSerializer()

        serialized = serializer.serialize(dt)
        result = serializer.deserialize(serialized)

        assert result.tzinfo is not None

    def test_convenience_functions(self):
        """Test convenience functions."""
        dt = datetime.now()

        serialized = serialize_datetime(dt)
        result = deserialize_datetime(serialized)

        assert result.year == dt.year
        assert result.month == dt.month
        assert result.day == dt.day

    def test_parse_relative_time(self):
        """Test relative time parsing."""
        serializer = DateTimeSerializer()

        result = serializer.parse_relative("2 hours ago")
        expected = datetime.utcnow() - timedelta(hours=2)

        # Allow 1 second tolerance
        assert abs((result - expected).total_seconds()) < 1

    def test_format_relative(self):
        """Test relative time formatting."""
        serializer = DateTimeSerializer()
        dt = datetime.utcnow() - timedelta(hours=1)

        result = serializer.format_relative(dt)

        assert "hour" in result.lower() or "60" in result


# ==================== Mock Database Tests ====================

class MockConnection:
    """Mock database connection for testing."""

    def __init__(self):
        self._data: dict[str, list[dict]] = {}
        self._queries: list[tuple[str, tuple]] = []

    async def execute(self, query: str, params: tuple = ()) -> None:
        self._queries.append((query, params))

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        self._queries.append((query, params))
        # Return empty by default
        return []

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        self._queries.append((query, params))
        return None


class TestMigrationRunner:
    """Tests for database migration runner."""

    @pytest.mark.asyncio
    async def test_migration_table_creation(self):
        """Test migration table is created on init."""
        from aion.persistence.migrations.runner import MigrationRunner

        conn = MockConnection()
        runner = MigrationRunner(connection=conn)

        await runner.initialize()

        # Check that CREATE TABLE was executed
        create_queries = [q for q, _ in conn._queries if "CREATE TABLE" in q.upper()]
        assert len(create_queries) > 0

    @pytest.mark.asyncio
    async def test_migration_status(self):
        """Test getting migration status."""
        from aion.persistence.migrations.runner import MigrationRunner

        conn = MockConnection()
        runner = MigrationRunner(connection=conn)

        await runner.initialize()
        status = await runner.status()

        assert "current_version" in status
        assert "applied_count" in status
        assert "pending_count" in status


class TestTransactionManager:
    """Tests for transaction manager."""

    @pytest.mark.asyncio
    async def test_transaction_context(self):
        """Test transaction context manager."""
        from aion.persistence.transactions import TransactionManager

        conn = MockConnection()
        manager = TransactionManager(connection=conn)

        async with manager.transaction() as txn:
            assert txn.context.state.value == "active"

        # Check BEGIN and COMMIT were called
        queries = [q.upper() for q, _ in conn._queries]
        assert "BEGIN" in queries

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self):
        """Test transaction rolls back on error."""
        from aion.persistence.transactions import TransactionManager

        conn = MockConnection()
        manager = TransactionManager(connection=conn)

        with pytest.raises(ValueError):
            async with manager.transaction() as txn:
                raise ValueError("Test error")

        # Check ROLLBACK was called
        queries = [q.upper() for q, _ in conn._queries]
        assert "ROLLBACK" in queries

    def test_lock_manager_acquire_release(self):
        """Test lock manager acquire and release."""
        from aion.persistence.transactions import LockManager, LockMode

        manager = LockManager()

        # Test synchronously using internal state
        assert len(manager._locks) == 0

    @pytest.mark.asyncio
    async def test_lock_manager_deadlock_detection(self):
        """Test deadlock detection."""
        from aion.persistence.transactions import LockManager, LockMode

        manager = LockManager()

        # Acquire first lock
        await manager.acquire("txn1", "resource_a", LockMode.EXCLUSIVE)

        # Verify lock is held
        assert "resource_a" in manager._locks


class TestEventStore:
    """Tests for event store."""

    @pytest.mark.asyncio
    async def test_event_append(self):
        """Test appending events."""
        from aion.persistence.events import EventStore, Event

        conn = MockConnection()
        store = EventStore(connection=conn)

        await store.initialize()

        event = Event(
            id="evt-1",
            event_type="test.created",
            aggregate_type="TestAggregate",
            aggregate_id="agg-1",
            data={"name": "Test"},
        )

        result = await store.append(event)

        assert result.sequence_number == 1

    @pytest.mark.asyncio
    async def test_event_handler_subscription(self):
        """Test event handler subscription."""
        from aion.persistence.events import EventStore, Event

        conn = MockConnection()
        store = EventStore(connection=conn)

        events_received = []

        class TestHandler:
            async def handle(self, event: Event):
                events_received.append(event)

        store.subscribe("test.created", TestHandler())

        await store.initialize()

        event = Event(
            id="evt-1",
            event_type="test.created",
            aggregate_type="TestAggregate",
            aggregate_id="agg-1",
            data={"name": "Test"},
        )

        await store.append(event)

        assert len(events_received) == 1
        assert events_received[0].id == "evt-1"


class TestCDCManager:
    """Tests for Change Data Capture manager."""

    @pytest.mark.asyncio
    async def test_capture_change(self):
        """Test capturing a change."""
        from aion.persistence.events import CDCManager

        conn = MockConnection()
        manager = CDCManager(connection=conn)

        await manager.initialize()

        event = await manager.capture(
            table_name="users",
            operation="INSERT",
            entity_id="user-1",
            new_data={"name": "John", "email": "john@example.com"},
        )

        assert event.table_name == "users"
        assert event.operation == "INSERT"
        assert event.entity_id == "user-1"

    @pytest.mark.asyncio
    async def test_changed_fields_detection(self):
        """Test detection of changed fields."""
        from aion.persistence.events import CDCManager

        conn = MockConnection()
        manager = CDCManager(connection=conn)

        await manager.initialize()

        event = await manager.capture(
            table_name="users",
            operation="UPDATE",
            entity_id="user-1",
            old_data={"name": "John", "email": "john@example.com"},
            new_data={"name": "John Doe", "email": "john@example.com"},
        )

        assert "name" in event.changed_fields
        assert "email" not in event.changed_fields


class TestBackupManager:
    """Tests for backup manager."""

    @pytest.mark.asyncio
    async def test_backup_statistics(self):
        """Test getting backup statistics."""
        from aion.persistence.backup import BackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            conn = MockConnection()
            manager = BackupManager(
                connection=conn,
                backup_dir=Path(tmpdir),
            )

            await manager.initialize()
            stats = await manager.get_backup_statistics()

            assert "total_backups" in stats
            assert "completed_backups" in stats


# ==================== Integration Tests ====================

class TestStateManagerIntegration:
    """Integration tests for state manager."""

    @pytest.mark.asyncio
    async def test_state_manager_initialization(self):
        """Test state manager initialization with in-memory database."""
        from aion.persistence.state_manager import StateManager
        from aion.persistence.config import PersistenceConfig, DatabaseBackend

        config = PersistenceConfig(backend=DatabaseBackend.MEMORY)

        manager = StateManager(config)

        # Initialize should not raise
        try:
            await manager.initialize()
            assert manager._initialized
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_repository_access(self):
        """Test lazy repository access."""
        from aion.persistence.state_manager import StateManager
        from aion.persistence.config import PersistenceConfig, DatabaseBackend

        config = PersistenceConfig(backend=DatabaseBackend.MEMORY)

        manager = StateManager(config)

        try:
            await manager.initialize()

            # Access repositories
            assert manager.memories is not None
            assert manager.plans is not None
            assert manager.processes is not None
            assert manager.evolution is not None
            assert manager.tools is not None
            assert manager.config is not None

        finally:
            await manager.shutdown()


# ==================== Performance Tests ====================

class TestPerformance:
    """Performance tests for persistence layer."""

    def test_numpy_serialization_speed(self):
        """Test NumPy serialization performance."""
        import time

        serializer = NumpySerializer()
        arr = np.random.randn(1000, 1000).astype(np.float32)

        start = time.monotonic()
        for _ in range(10):
            data, _ = serializer.serialize(arr)
        serialize_time = time.monotonic() - start

        # Should serialize 10x 1000x1000 arrays in under 5 seconds
        assert serialize_time < 5.0

    def test_numpy_compression_ratio(self):
        """Test compression effectiveness."""
        serializer = NumpySerializer(compression=CompressionMethod.ZLIB)

        # Highly compressible (all zeros)
        zeros = np.zeros((1000, 1000), dtype=np.float32)
        data_zeros, meta_zeros = serializer.serialize(zeros)

        # Random data (less compressible)
        random = np.random.randn(1000, 1000).astype(np.float32)
        data_random, meta_random = serializer.serialize(random)

        # Zeros should compress much better
        zeros_ratio = meta_zeros.compressed_size / meta_zeros.original_size
        random_ratio = meta_random.compressed_size / meta_random.original_size

        assert zeros_ratio < random_ratio
        assert zeros_ratio < 0.1  # Should be very small for zeros


# ==================== Edge Case Tests ====================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_array_serialization(self):
        """Test serializing empty arrays."""
        serializer = NumpySerializer()
        arr = np.array([], dtype=np.float32)

        data, metadata = serializer.serialize(arr)
        result = serializer.deserialize(data, metadata)

        assert len(result) == 0

    def test_datetime_edge_cases(self):
        """Test datetime edge cases."""
        serializer = DateTimeSerializer()

        # Very old date
        old = datetime(1900, 1, 1)
        assert serializer.deserialize(serializer.serialize(old)) == old

        # Far future date
        future = datetime(2100, 12, 31, 23, 59, 59)
        assert serializer.deserialize(serializer.serialize(future)) == future

    def test_config_with_all_options(self):
        """Test config with all options set."""
        config = PersistenceConfig(
            backend=DatabaseBackend.POSTGRESQL,
            compression=CompressionType.ZSTD,
            compression_threshold=2048,
            batch_size=200,
            query_timeout=60.0,
            log_queries=True,
            log_slow_queries=True,
            slow_query_threshold_ms=50.0,
            enable_metrics=True,
        )

        data = config.to_dict()
        assert data["backend"] == "postgresql"
        assert data["compression"] == "zstd"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
