"""
AION Process Repository

Persistence for process and agent state including:
- Process lifecycle tracking
- Agent configurations
- Scheduled tasks
- Events and message history
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
import uuid

import structlog

from aion.persistence.repositories.base import BaseRepository, QueryOptions
from aion.persistence.database import DatabaseManager
from aion.persistence.backends.redis_cache import CacheManager

logger = structlog.get_logger(__name__)

# Import process types
try:
    from aion.systems.process.models import (
        ProcessInfo,
        ProcessState,
        ProcessPriority,
        ProcessType,
        RestartPolicy,
        ResourceLimits,
        ResourceUsage,
        TaskDefinition,
        Event,
        ProcessCheckpoint,
    )
except ImportError:
    # Fallback types
    from enum import Enum
    from dataclasses import dataclass, field

    class ProcessState(Enum):
        CREATED = "created"
        STARTING = "starting"
        RUNNING = "running"
        PAUSED = "paused"
        STOPPING = "stopping"
        STOPPED = "stopped"
        FAILED = "failed"
        TERMINATED = "terminated"

    class ProcessPriority(Enum):
        CRITICAL = 0
        HIGH = 1
        NORMAL = 2
        LOW = 3
        IDLE = 4

    class ProcessType(Enum):
        AGENT = "agent"
        TASK = "task"
        WORKER = "worker"
        SYSTEM = "system"
        SCHEDULED = "scheduled"
        CHILD = "child"

    class RestartPolicy(Enum):
        NEVER = "never"
        ON_FAILURE = "on_failure"
        ALWAYS = "always"
        EXPONENTIAL_BACKOFF = "exponential_backoff"

    @dataclass
    class ResourceLimits:
        max_memory_mb: Optional[int] = None
        max_cpu_percent: Optional[float] = None
        max_tokens_per_minute: Optional[int] = None
        max_tokens_total: Optional[int] = None
        max_runtime_seconds: Optional[int] = None

        def to_dict(self):
            return vars(self)

        @classmethod
        def from_dict(cls, data):
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @dataclass
    class ResourceUsage:
        memory_mb: float = 0.0
        cpu_percent: float = 0.0
        tokens_used: int = 0
        runtime_seconds: float = 0.0

    @dataclass
    class ProcessInfo:
        id: str
        name: str
        type: ProcessType
        state: ProcessState
        priority: ProcessPriority
        created_at: datetime
        started_at: Optional[datetime] = None
        stopped_at: Optional[datetime] = None
        parent_id: Optional[str] = None
        restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
        restart_count: int = 0
        max_restarts: int = 5
        limits: ResourceLimits = field(default_factory=ResourceLimits)
        usage: ResourceUsage = field(default_factory=ResourceUsage)
        error: Optional[str] = None
        exit_code: Optional[int] = None
        metadata: dict = field(default_factory=dict)

    @dataclass
    class TaskDefinition:
        id: str
        name: str
        handler: str
        params: dict = field(default_factory=dict)
        schedule_type: str = "once"
        run_at: Optional[datetime] = None
        interval_seconds: Optional[int] = None
        cron_expression: Optional[str] = None
        enabled: bool = True
        last_run: Optional[datetime] = None
        next_run: Optional[datetime] = None
        run_count: int = 0
        failure_count: int = 0
        priority: ProcessPriority = ProcessPriority.NORMAL
        limits: ResourceLimits = field(default_factory=ResourceLimits)

    @dataclass
    class Event:
        id: str
        type: str
        source: str
        payload: dict
        timestamp: datetime = field(default_factory=datetime.now)
        correlation_id: Optional[str] = None

    @dataclass
    class ProcessCheckpoint:
        id: str
        process_id: str
        timestamp: datetime
        state: ProcessState
        internal_state: dict = field(default_factory=dict)
        resource_usage: ResourceUsage = field(default_factory=ResourceUsage)


class ProcessRepository(BaseRepository[ProcessInfo]):
    """
    Repository for process persistence.

    Features:
    - Full process lifecycle tracking
    - Parent-child relationships
    - Resource usage snapshots
    - Restart tracking
    """

    def __init__(
        self,
        db: DatabaseManager,
        cache: Optional[CacheManager] = None,
    ):
        super().__init__(db, cache)
        self._table_name = "processes"
        self._soft_delete_column = None

    def _serialize(self, process: ProcessInfo) -> dict[str, Any]:
        """Serialize ProcessInfo to database row."""
        limits = process.limits.to_dict() if hasattr(process.limits, 'to_dict') else vars(process.limits)
        usage = process.usage.to_dict() if hasattr(process.usage, 'to_dict') else vars(process.usage)

        return {
            "id": process.id,
            "name": process.name,
            "type": process.type.value if hasattr(process.type, 'value') else str(process.type),
            "state": process.state.value if hasattr(process.state, 'value') else str(process.state),
            "priority": process.priority.value if hasattr(process.priority, 'value') else process.priority,
            "config": self._to_json(process.metadata.get("config", {})),
            "created_at": self._from_datetime(process.created_at),
            "started_at": self._from_datetime(process.started_at),
            "stopped_at": self._from_datetime(process.stopped_at),
            "parent_id": process.parent_id,
            "restart_policy": process.restart_policy.value if hasattr(process.restart_policy, 'value') else str(process.restart_policy),
            "restart_count": process.restart_count,
            "max_restarts": process.max_restarts,
            "resource_limits": self._to_json(limits),
            "resource_usage": self._to_json(usage),
            "internal_state": self._to_json(process.metadata.get("internal_state", {})),
            "error": process.error,
            "exit_code": process.exit_code,
            "metadata": self._to_json(process.metadata),
        }

    def _deserialize(self, row: dict[str, Any]) -> ProcessInfo:
        """Deserialize database row to ProcessInfo."""
        limits_data = self._from_json(row.get("resource_limits")) or {}
        usage_data = self._from_json(row.get("resource_usage")) or {}
        metadata = self._from_json(row.get("metadata")) or {}

        return ProcessInfo(
            id=row["id"],
            name=row["name"],
            type=ProcessType(row["type"]),
            state=ProcessState(row["state"]),
            priority=ProcessPriority(row["priority"]),
            created_at=self._to_datetime(row.get("created_at")) or datetime.now(),
            started_at=self._to_datetime(row.get("started_at")),
            stopped_at=self._to_datetime(row.get("stopped_at")),
            parent_id=row.get("parent_id"),
            restart_policy=RestartPolicy(row.get("restart_policy", "on_failure")),
            restart_count=row.get("restart_count", 0),
            max_restarts=row.get("max_restarts", 5),
            limits=ResourceLimits.from_dict(limits_data) if hasattr(ResourceLimits, 'from_dict') else ResourceLimits(**limits_data),
            usage=ResourceUsage(**usage_data),
            error=row.get("error"),
            exit_code=row.get("exit_code"),
            metadata=metadata,
        )

    # === State Management ===

    async def update_state(
        self,
        id: str,
        state: ProcessState,
        error: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> bool:
        """Update process state."""
        fields = {
            "state": state.value if hasattr(state, 'value') else str(state),
        }

        if state == ProcessState.RUNNING:
            fields["started_at"] = datetime.now().isoformat()
        elif state in (ProcessState.STOPPED, ProcessState.FAILED, ProcessState.TERMINATED):
            fields["stopped_at"] = datetime.now().isoformat()

        if error is not None:
            fields["error"] = error

        if exit_code is not None:
            fields["exit_code"] = exit_code

        return await self.update_fields(id, fields)

    async def update_resource_usage(
        self,
        id: str,
        usage: ResourceUsage,
    ) -> bool:
        """Update resource usage snapshot."""
        usage_data = usage.to_dict() if hasattr(usage, 'to_dict') else vars(usage)
        return await self.update_fields(id, {
            "resource_usage": self._to_json(usage_data),
        })

    async def save_internal_state(
        self,
        id: str,
        state: dict[str, Any],
    ) -> bool:
        """Save agent internal state."""
        return await self.update_fields(id, {
            "internal_state": self._to_json(state),
        })

    async def increment_restart_count(self, id: str) -> int:
        """Increment restart count and return new value."""
        query = "UPDATE processes SET restart_count = restart_count + 1 WHERE id = ?"
        await self.db.execute(query, (id,))

        row = await self.db.fetch_one(
            "SELECT restart_count FROM processes WHERE id = ?",
            (id,),
        )
        return row["restart_count"] if row else 0

    # === Query Helpers ===

    async def find_by_state(
        self,
        state: ProcessState,
        options: Optional[QueryOptions] = None,
    ) -> list[ProcessInfo]:
        """Find processes by state."""
        state_value = state.value if hasattr(state, 'value') else str(state)
        return await self.find_where("state = ?", (state_value,), options=options)

    async def find_by_type(
        self,
        process_type: ProcessType,
        options: Optional[QueryOptions] = None,
    ) -> list[ProcessInfo]:
        """Find processes by type."""
        type_value = process_type.value if hasattr(process_type, 'value') else str(process_type)
        return await self.find_where("type = ?", (type_value,), options=options)

    async def find_running(
        self,
        options: Optional[QueryOptions] = None,
    ) -> list[ProcessInfo]:
        """Find running processes."""
        return await self.find_by_state(ProcessState.RUNNING, options)

    async def find_restartable(self) -> list[ProcessInfo]:
        """Find processes that should be restarted on startup."""
        query = """
            SELECT * FROM processes
            WHERE state = 'running'
            AND restart_policy != 'never'
            AND restart_count < max_restarts
        """
        rows = await self.db.fetch_all(query)
        return [self._deserialize(row) for row in rows]

    async def find_children(
        self,
        parent_id: str,
        options: Optional[QueryOptions] = None,
    ) -> list[ProcessInfo]:
        """Find child processes."""
        return await self.find_where("parent_id = ?", (parent_id,), options=options)

    async def find_agents(
        self,
        options: Optional[QueryOptions] = None,
    ) -> list[ProcessInfo]:
        """Find all agent processes."""
        return await self.find_by_type(ProcessType.AGENT, options)

    # === Checkpoint Operations ===

    async def save_checkpoint(
        self,
        checkpoint: ProcessCheckpoint,
    ) -> str:
        """Save a process checkpoint."""
        checkpoint_id = checkpoint.id or str(uuid.uuid4())
        usage = checkpoint.resource_usage.to_dict() if hasattr(checkpoint.resource_usage, 'to_dict') else vars(checkpoint.resource_usage)

        query = """
            INSERT INTO process_checkpoints (id, process_id, timestamp, state, internal_state, resource_usage, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        await self.db.execute(query, (
            checkpoint_id,
            checkpoint.process_id,
            checkpoint.timestamp.isoformat(),
            checkpoint.state.value if hasattr(checkpoint.state, 'value') else str(checkpoint.state),
            self._to_json(checkpoint.internal_state),
            self._to_json(usage),
            self._to_json({}),
        ))

        return checkpoint_id

    async def get_checkpoints(
        self,
        process_id: str,
        limit: int = 10,
    ) -> list[ProcessCheckpoint]:
        """Get checkpoints for a process."""
        query = """
            SELECT * FROM process_checkpoints
            WHERE process_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """

        rows = await self.db.fetch_all(query, (process_id, limit))

        return [
            ProcessCheckpoint(
                id=row["id"],
                process_id=row["process_id"],
                timestamp=self._to_datetime(row["timestamp"]) or datetime.now(),
                state=ProcessState(row["state"]),
                internal_state=self._from_json(row.get("internal_state")) or {},
                resource_usage=ResourceUsage(**(self._from_json(row.get("resource_usage")) or {})),
            )
            for row in rows
        ]

    async def get_statistics(self) -> dict[str, Any]:
        """Get process system statistics."""
        query = """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN state = 'running' THEN 1 END) as running,
                COUNT(CASE WHEN state = 'stopped' THEN 1 END) as stopped,
                COUNT(CASE WHEN state = 'failed' THEN 1 END) as failed,
                COUNT(CASE WHEN type = 'agent' THEN 1 END) as agents,
                COUNT(CASE WHEN type = 'task' THEN 1 END) as tasks,
                SUM(restart_count) as total_restarts
            FROM processes
        """

        stats = await self.db.fetch_one(query)

        return {
            "total": stats["total"] if stats else 0,
            "by_state": {
                "running": stats["running"] if stats else 0,
                "stopped": stats["stopped"] if stats else 0,
                "failed": stats["failed"] if stats else 0,
            },
            "by_type": {
                "agents": stats["agents"] if stats else 0,
                "tasks": stats["tasks"] if stats else 0,
            },
            "total_restarts": stats["total_restarts"] if stats else 0,
        }


class TaskRepository(BaseRepository[TaskDefinition]):
    """Repository for scheduled task persistence."""

    def __init__(
        self,
        db: DatabaseManager,
        cache: Optional[CacheManager] = None,
    ):
        super().__init__(db, cache)
        self._table_name = "scheduled_tasks"
        self._soft_delete_column = None

    def _serialize(self, task: TaskDefinition) -> dict[str, Any]:
        """Serialize TaskDefinition to database row."""
        limits = task.limits.to_dict() if hasattr(task.limits, 'to_dict') else vars(task.limits)

        return {
            "id": task.id,
            "name": task.name,
            "handler": task.handler,
            "params": self._to_json(task.params),
            "schedule_type": task.schedule_type,
            "run_at": self._from_datetime(task.run_at),
            "interval_seconds": task.interval_seconds,
            "cron_expression": task.cron_expression,
            "enabled": 1 if task.enabled else 0,
            "last_run": self._from_datetime(task.last_run),
            "next_run": self._from_datetime(task.next_run),
            "run_count": task.run_count,
            "failure_count": task.failure_count,
            "priority": task.priority.value if hasattr(task.priority, 'value') else task.priority,
            "resource_limits": self._to_json(limits),
        }

    def _deserialize(self, row: dict[str, Any]) -> TaskDefinition:
        """Deserialize database row to TaskDefinition."""
        limits_data = self._from_json(row.get("resource_limits")) or {}

        return TaskDefinition(
            id=row["id"],
            name=row["name"],
            handler=row["handler"],
            params=self._from_json(row.get("params")) or {},
            schedule_type=row.get("schedule_type", "once"),
            run_at=self._to_datetime(row.get("run_at")),
            interval_seconds=row.get("interval_seconds"),
            cron_expression=row.get("cron_expression"),
            enabled=bool(row.get("enabled", 1)),
            last_run=self._to_datetime(row.get("last_run")),
            next_run=self._to_datetime(row.get("next_run")),
            run_count=row.get("run_count", 0),
            failure_count=row.get("failure_count", 0),
            priority=ProcessPriority(row.get("priority", 2)),
            limits=ResourceLimits.from_dict(limits_data) if hasattr(ResourceLimits, 'from_dict') else ResourceLimits(**limits_data),
        )

    async def find_pending(self) -> list[TaskDefinition]:
        """Find tasks ready to run."""
        query = """
            SELECT * FROM scheduled_tasks
            WHERE enabled = 1 AND next_run <= ?
            ORDER BY priority ASC, next_run ASC
        """
        rows = await self.db.fetch_all(query, (datetime.now().isoformat(),))
        return [self._deserialize(row) for row in rows]

    async def find_enabled(
        self,
        options: Optional[QueryOptions] = None,
    ) -> list[TaskDefinition]:
        """Find all enabled tasks."""
        return await self.find_where("enabled = 1", (), options=options)

    async def update_execution(
        self,
        id: str,
        next_run: Optional[datetime],
        success: bool,
    ) -> bool:
        """Update task after execution."""
        if success:
            query = """
                UPDATE scheduled_tasks
                SET last_run = ?, next_run = ?, run_count = run_count + 1
                WHERE id = ?
            """
        else:
            query = """
                UPDATE scheduled_tasks
                SET last_run = ?, next_run = ?, failure_count = failure_count + 1
                WHERE id = ?
            """

        await self.db.execute(query, (
            datetime.now().isoformat(),
            next_run.isoformat() if next_run else None,
            id,
        ))

        if self.cache:
            await self.cache.delete(self._cache_key(id))

        return True

    async def set_enabled(self, id: str, enabled: bool) -> bool:
        """Enable or disable a task."""
        return await self.update_fields(id, {"enabled": 1 if enabled else 0})


class EventRepository:
    """Repository for event persistence."""

    def __init__(
        self,
        db: DatabaseManager,
    ):
        self.db = db

    async def save(self, event: Event) -> str:
        """Save an event."""
        event_id = event.id or str(uuid.uuid4())

        query = """
            INSERT INTO events (id, type, source, payload, correlation_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        payload = event.payload if isinstance(event.payload, str) else json.dumps(event.payload, default=str)

        await self.db.execute(query, (
            event_id,
            event.type,
            event.source,
            payload,
            event.correlation_id,
            event.timestamp.isoformat(),
        ))

        return event_id

    async def get(self, id: str) -> Optional[Event]:
        """Get an event by ID."""
        query = "SELECT * FROM events WHERE id = ?"
        row = await self.db.fetch_one(query, (id,))

        if not row:
            return None

        payload = row.get("payload", "{}")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {"raw": payload}

        return Event(
            id=row["id"],
            type=row["type"],
            source=row["source"],
            payload=payload,
            timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.now(),
            correlation_id=row.get("correlation_id"),
        )

    async def find_by_type(
        self,
        event_type: str,
        limit: int = 100,
    ) -> list[Event]:
        """Find events by type."""
        query = """
            SELECT * FROM events
            WHERE type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """

        rows = await self.db.fetch_all(query, (event_type, limit))
        events = []

        for row in rows:
            payload = row.get("payload", "{}")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {"raw": payload}

            events.append(Event(
                id=row["id"],
                type=row["type"],
                source=row["source"],
                payload=payload,
                timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.now(),
                correlation_id=row.get("correlation_id"),
            ))

        return events

    async def find_by_correlation(
        self,
        correlation_id: str,
    ) -> list[Event]:
        """Find events by correlation ID."""
        query = """
            SELECT * FROM events
            WHERE correlation_id = ?
            ORDER BY timestamp ASC
        """

        rows = await self.db.fetch_all(query, (correlation_id,))
        events = []

        for row in rows:
            payload = row.get("payload", "{}")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {"raw": payload}

            events.append(Event(
                id=row["id"],
                type=row["type"],
                source=row["source"],
                payload=payload,
                timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.now(),
                correlation_id=row.get("correlation_id"),
            ))

        return events

    async def cleanup(
        self,
        older_than_days: int = 30,
    ) -> int:
        """Delete old events."""
        query = """
            DELETE FROM events
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """
        await self.db.execute(query, (older_than_days,))
        return 0


# Need json import for EventRepository
import json
