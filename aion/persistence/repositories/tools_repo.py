"""
AION Tools Repository

Persistence for tool execution history and patterns:
- Tool execution logs
- Learned usage patterns
- Tool performance analytics
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


from dataclasses import dataclass, field


@dataclass
class ToolExecution:
    """Record of a tool execution."""
    id: str
    tool_name: str
    params: dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    success: bool = True
    latency_ms: float = 0.0
    tokens_used: int = 0
    process_id: Optional[str] = None
    plan_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class ToolPattern:
    """Learned tool usage pattern."""
    id: int
    task_pattern: str
    tool_sequence: list[str]
    success_rate: float
    avg_latency_ms: float
    sample_count: int
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ToolsRepository(BaseRepository[ToolExecution]):
    """
    Repository for tool execution history.

    Features:
    - Execution logging
    - Pattern learning
    - Performance analytics
    """

    def __init__(
        self,
        db: DatabaseManager,
        cache: Optional[CacheManager] = None,
    ):
        super().__init__(db, cache)
        self._table_name = "tool_executions"
        self._soft_delete_column = None

    def _serialize(self, execution: ToolExecution) -> dict[str, Any]:
        """Serialize ToolExecution to database row."""
        result_json = None
        if execution.result is not None:
            try:
                result_json = self._to_json(execution.result)
            except Exception:
                result_json = str(execution.result)[:1000]

        return {
            "id": execution.id,
            "tool_name": execution.tool_name,
            "params": self._to_json(execution.params),
            "result": result_json,
            "error": execution.error,
            "success": 1 if execution.success else 0,
            "latency_ms": execution.latency_ms,
            "tokens_used": execution.tokens_used,
            "process_id": execution.process_id,
            "plan_id": execution.plan_id,
            "started_at": self._from_datetime(execution.started_at),
            "completed_at": self._from_datetime(execution.completed_at),
        }

    def _deserialize(self, row: dict[str, Any]) -> ToolExecution:
        """Deserialize database row to ToolExecution."""
        result = self._from_json(row.get("result"))

        return ToolExecution(
            id=row["id"],
            tool_name=row["tool_name"],
            params=self._from_json(row.get("params")) or {},
            result=result,
            error=row.get("error"),
            success=bool(row.get("success", 1)),
            latency_ms=row.get("latency_ms", 0.0),
            tokens_used=row.get("tokens_used", 0),
            process_id=row.get("process_id"),
            plan_id=row.get("plan_id"),
            started_at=self._to_datetime(row.get("started_at")) or datetime.now(),
            completed_at=self._to_datetime(row.get("completed_at")),
        )

    async def log_execution(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: Any = None,
        error: Optional[str] = None,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        process_id: Optional[str] = None,
        plan_id: Optional[str] = None,
    ) -> str:
        """Log a tool execution."""
        execution = ToolExecution(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            params=params,
            result=result,
            error=error,
            success=error is None,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            process_id=process_id,
            plan_id=plan_id,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

        return await self.create(execution)

    async def find_by_tool(
        self,
        tool_name: str,
        options: Optional[QueryOptions] = None,
    ) -> list[ToolExecution]:
        """Find executions by tool name."""
        return await self.find_where(
            "tool_name = ?",
            (tool_name,),
            options=options,
        )

    async def find_by_process(
        self,
        process_id: str,
        options: Optional[QueryOptions] = None,
    ) -> list[ToolExecution]:
        """Find executions by process."""
        return await self.find_where(
            "process_id = ?",
            (process_id,),
            options=options,
        )

    async def find_by_plan(
        self,
        plan_id: str,
        options: Optional[QueryOptions] = None,
    ) -> list[ToolExecution]:
        """Find executions by plan."""
        return await self.find_where(
            "plan_id = ?",
            (plan_id,),
            options=options,
        )

    async def find_failures(
        self,
        options: Optional[QueryOptions] = None,
    ) -> list[ToolExecution]:
        """Find failed executions."""
        return await self.find_where(
            "success = 0",
            (),
            options=options,
        )

    # === Pattern Operations ===

    async def save_pattern(
        self,
        task_pattern: str,
        tool_sequence: list[str],
        success: bool,
        latency_ms: float,
    ) -> None:
        """Save or update a tool usage pattern."""
        # Check if pattern exists
        query = """
            SELECT id, success_rate, avg_latency_ms, sample_count
            FROM tool_patterns
            WHERE task_pattern = ?
        """
        existing = await self.db.fetch_one(query, (task_pattern,))

        if existing:
            # Update existing pattern
            old_count = existing["sample_count"]
            new_count = old_count + 1

            # Calculate new averages
            new_success_rate = (
                (existing["success_rate"] * old_count + (1.0 if success else 0.0))
                / new_count
            )
            new_avg_latency = (
                (existing["avg_latency_ms"] * old_count + latency_ms)
                / new_count
            )

            update_query = """
                UPDATE tool_patterns
                SET tool_sequence = ?, success_rate = ?, avg_latency_ms = ?,
                    sample_count = ?, updated_at = ?
                WHERE id = ?
            """
            await self.db.execute(update_query, (
                self._to_json(tool_sequence),
                new_success_rate,
                new_avg_latency,
                new_count,
                datetime.now().isoformat(),
                existing["id"],
            ))
        else:
            # Create new pattern
            insert_query = """
                INSERT INTO tool_patterns (task_pattern, tool_sequence, success_rate, avg_latency_ms, sample_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            await self.db.execute(insert_query, (
                task_pattern,
                self._to_json(tool_sequence),
                1.0 if success else 0.0,
                latency_ms,
                1,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))

    async def find_matching_patterns(
        self,
        task_description: str,
        min_success_rate: float = 0.5,
        limit: int = 5,
    ) -> list[ToolPattern]:
        """Find patterns that might match a task."""
        # Simple LIKE matching - could be enhanced with embeddings
        query = """
            SELECT * FROM tool_patterns
            WHERE task_pattern LIKE ? AND success_rate >= ?
            ORDER BY sample_count DESC, success_rate DESC
            LIMIT ?
        """

        # Extract keywords from task
        keywords = task_description.lower().split()[:3]
        pattern = "%".join(keywords)

        rows = await self.db.fetch_all(query, (f"%{pattern}%", min_success_rate, limit))

        return [
            ToolPattern(
                id=row["id"],
                task_pattern=row["task_pattern"],
                tool_sequence=self._from_json(row.get("tool_sequence")) or [],
                success_rate=row.get("success_rate", 0.0),
                avg_latency_ms=row.get("avg_latency_ms", 0.0),
                sample_count=row.get("sample_count", 0),
                created_at=self._to_datetime(row.get("created_at")) or datetime.now(),
                updated_at=self._to_datetime(row.get("updated_at")) or datetime.now(),
            )
            for row in rows
        ]

    async def get_best_pattern_for_task(
        self,
        task_description: str,
    ) -> Optional[ToolPattern]:
        """Get the best matching pattern for a task."""
        patterns = await self.find_matching_patterns(
            task_description,
            min_success_rate=0.7,
            limit=1,
        )
        return patterns[0] if patterns else None

    # === Analytics ===

    async def get_tool_statistics(
        self,
        tool_name: Optional[str] = None,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Get tool execution statistics."""
        if tool_name:
            query = """
                SELECT
                    tool_name,
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(latency_ms) as avg_latency,
                    MIN(latency_ms) as min_latency,
                    MAX(latency_ms) as max_latency,
                    SUM(tokens_used) as total_tokens
                FROM tool_executions
                WHERE tool_name = ? AND started_at >= datetime('now', '-' || ? || ' hours')
                GROUP BY tool_name
            """
            row = await self.db.fetch_one(query, (tool_name, hours))
        else:
            query = """
                SELECT
                    'all' as tool_name,
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(latency_ms) as avg_latency,
                    MIN(latency_ms) as min_latency,
                    MAX(latency_ms) as max_latency,
                    SUM(tokens_used) as total_tokens
                FROM tool_executions
                WHERE started_at >= datetime('now', '-' || ? || ' hours')
            """
            row = await self.db.fetch_one(query, (hours,))

        if not row or not row["total"]:
            return {
                "tool_name": tool_name or "all",
                "period_hours": hours,
                "total_executions": 0,
            }

        return {
            "tool_name": row["tool_name"],
            "period_hours": hours,
            "total_executions": row["total"],
            "successful": row["successful"],
            "failed": row["failed"],
            "success_rate": row["successful"] / row["total"] if row["total"] > 0 else 0,
            "latency_ms": {
                "avg": row["avg_latency"],
                "min": row["min_latency"],
                "max": row["max_latency"],
            },
            "total_tokens": row["total_tokens"],
        }

    async def get_top_tools(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get most used tools."""
        query = """
            SELECT
                tool_name,
                COUNT(*) as count,
                AVG(latency_ms) as avg_latency,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
            FROM tool_executions
            WHERE started_at >= datetime('now', '-' || ? || ' hours')
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT ?
        """

        rows = await self.db.fetch_all(query, (hours, limit))

        return [
            {
                "tool_name": row["tool_name"],
                "count": row["count"],
                "avg_latency_ms": row["avg_latency"],
                "success_rate": row["success_rate"],
            }
            for row in rows
        ]

    async def get_slowest_tools(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get slowest tools by average latency."""
        query = """
            SELECT
                tool_name,
                COUNT(*) as count,
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency
            FROM tool_executions
            WHERE started_at >= datetime('now', '-' || ? || ' hours')
            GROUP BY tool_name
            HAVING count >= 5
            ORDER BY avg_latency DESC
            LIMIT ?
        """

        rows = await self.db.fetch_all(query, (hours, limit))

        return [
            {
                "tool_name": row["tool_name"],
                "count": row["count"],
                "avg_latency_ms": row["avg_latency"],
                "max_latency_ms": row["max_latency"],
            }
            for row in rows
        ]

    async def cleanup_old_executions(
        self,
        days: int = 30,
    ) -> int:
        """Delete old execution records."""
        query = """
            DELETE FROM tool_executions
            WHERE started_at < datetime('now', '-' || ? || ' days')
        """
        await self.db.execute(query, (days,))
        return 0

    async def get_global_statistics(self) -> dict[str, Any]:
        """Get global tool statistics."""
        execution_query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                COUNT(DISTINCT tool_name) as unique_tools,
                SUM(tokens_used) as total_tokens,
                AVG(latency_ms) as avg_latency
            FROM tool_executions
        """

        pattern_query = """
            SELECT
                COUNT(*) as total,
                AVG(success_rate) as avg_success_rate,
                SUM(sample_count) as total_samples
            FROM tool_patterns
        """

        executions = await self.db.fetch_one(execution_query)
        patterns = await self.db.fetch_one(pattern_query)

        return {
            "executions": {
                "total": executions["total"] if executions else 0,
                "successful": executions["successful"] if executions else 0,
                "unique_tools": executions["unique_tools"] if executions else 0,
                "total_tokens": executions["total_tokens"] if executions else 0,
                "avg_latency_ms": executions["avg_latency"] if executions else 0,
                "success_rate": (
                    executions["successful"] / executions["total"]
                    if executions and executions["total"] > 0 else 0
                ),
            },
            "patterns": {
                "total": patterns["total"] if patterns else 0,
                "avg_success_rate": patterns["avg_success_rate"] if patterns else 0,
                "total_samples": patterns["total_samples"] if patterns else 0,
            },
        }
