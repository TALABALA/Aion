"""
AION Execution History Manager

History tracking for workflow executions.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import structlog

from aion.automation.types import (
    WorkflowExecution,
    ExecutionStatus,
    StepResult,
)

logger = structlog.get_logger(__name__)


class ExecutionHistoryManager:
    """
    Manages execution history for analytics and debugging.

    Features:
    - Execution history storage
    - Analytics and metrics
    - Query and filtering
    - Retention management
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        max_history_count: int = 10000,
        retention_days: int = 30,
    ):
        self.persistence_path = persistence_path
        self.max_history_count = max_history_count
        self.retention_days = retention_days

        # History storage
        self._history: Dict[str, ExecutionRecord] = {}

        # Indices
        self._by_workflow: Dict[str, List[str]] = defaultdict(list)
        self._by_status: Dict[ExecutionStatus, List[str]] = defaultdict(list)
        self._by_date: Dict[str, List[str]] = defaultdict(list)  # YYYY-MM-DD -> ids

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the history manager."""
        if self._initialized:
            return

        # Load persisted history
        if self.persistence_path:
            await self._load_history()

        self._initialized = True
        logger.info("Execution history manager initialized", records=len(self._history))

    async def shutdown(self) -> None:
        """Shutdown the history manager."""
        if self.persistence_path:
            await self._save_history()

        self._initialized = False

    # === History Operations ===

    async def record(
        self,
        execution: WorkflowExecution,
    ) -> str:
        """Record an execution in history."""
        async with self._lock:
            record = ExecutionRecord(
                id=execution.id,
                workflow_id=execution.workflow_id,
                workflow_name=execution.workflow_name,
                workflow_version=execution.workflow_version,
                trigger_type=execution.trigger_type.value if execution.trigger_type else None,
                status=execution.status,
                inputs=execution.inputs,
                outputs=execution.outputs,
                error=execution.error,
                step_count=len(execution.step_results),
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_ms=execution.duration_ms,
                initiated_by=execution.initiated_by,
            )

            self._history[record.id] = record

            # Update indices
            self._by_workflow[record.workflow_id].append(record.id)
            self._by_status[record.status].append(record.id)
            if record.started_at:
                date_key = record.started_at.strftime("%Y-%m-%d")
                self._by_date[date_key].append(record.id)

            # Enforce limits
            await self._enforce_limits()

            return record.id

    async def get(
        self,
        execution_id: str,
    ) -> Optional[ExecutionRecord]:
        """Get a history record by ID."""
        return self._history.get(execution_id)

    async def list(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ExecutionRecord]:
        """List history records with filters."""
        # Start with all records or filtered by workflow
        if workflow_id:
            record_ids = set(self._by_workflow.get(workflow_id, []))
        else:
            record_ids = set(self._history.keys())

        # Filter by status
        if status:
            status_ids = set(self._by_status.get(status, []))
            record_ids &= status_ids

        # Get records
        records = [self._history[rid] for rid in record_ids if rid in self._history]

        # Filter by date
        if from_date:
            records = [r for r in records if r.started_at and r.started_at >= from_date]
        if to_date:
            records = [r for r in records if r.started_at and r.started_at <= to_date]

        # Sort by started_at descending
        records.sort(key=lambda r: r.started_at or datetime.min, reverse=True)

        # Paginate
        return records[offset:offset + limit]

    async def delete(
        self,
        execution_id: str,
    ) -> bool:
        """Delete a history record."""
        async with self._lock:
            record = self._history.pop(execution_id, None)
            if not record:
                return False

            # Update indices
            if record.workflow_id in self._by_workflow:
                if execution_id in self._by_workflow[record.workflow_id]:
                    self._by_workflow[record.workflow_id].remove(execution_id)

            if record.status in self._by_status:
                if execution_id in self._by_status[record.status]:
                    self._by_status[record.status].remove(execution_id)

            return True

    # === Analytics ===

    async def get_workflow_stats(
        self,
        workflow_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get statistics for a workflow."""
        cutoff = datetime.now() - timedelta(days=days)
        records = await self.list(workflow_id=workflow_id, from_date=cutoff, limit=10000)

        if not records:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
            }

        total = len(records)
        completed = len([r for r in records if r.status == ExecutionStatus.COMPLETED])
        failed = len([r for r in records if r.status == ExecutionStatus.FAILED])

        durations = [r.duration_ms for r in records if r.duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_executions": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0.0,
            "avg_duration_ms": avg_duration,
            "min_duration_ms": min(durations) if durations else 0.0,
            "max_duration_ms": max(durations) if durations else 0.0,
        }

    async def get_daily_stats(
        self,
        days: int = 30,
        workflow_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get daily execution statistics."""
        stats = []
        today = datetime.now().date()

        for i in range(days):
            date = today - timedelta(days=i)
            date_key = date.strftime("%Y-%m-%d")

            record_ids = self._by_date.get(date_key, [])
            records = [self._history[rid] for rid in record_ids if rid in self._history]

            if workflow_id:
                records = [r for r in records if r.workflow_id == workflow_id]

            if records:
                completed = len([r for r in records if r.status == ExecutionStatus.COMPLETED])
                failed = len([r for r in records if r.status == ExecutionStatus.FAILED])
                durations = [r.duration_ms for r in records if r.duration_ms > 0]

                stats.append({
                    "date": date_key,
                    "total": len(records),
                    "completed": completed,
                    "failed": failed,
                    "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
                })
            else:
                stats.append({
                    "date": date_key,
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "avg_duration_ms": 0.0,
                })

        return stats

    async def get_error_analysis(
        self,
        workflow_id: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Analyze errors in executions."""
        cutoff = datetime.now() - timedelta(days=days)
        records = await self.list(
            workflow_id=workflow_id,
            status=ExecutionStatus.FAILED,
            from_date=cutoff,
            limit=1000,
        )

        error_counts: Dict[str, int] = defaultdict(int)
        for record in records:
            error_key = record.error[:100] if record.error else "Unknown error"
            error_counts[error_key] += 1

        # Sort by count
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_failures": len(records),
            "unique_errors": len(error_counts),
            "top_errors": [
                {"error": e, "count": c}
                for e, c in sorted_errors[:10]
            ],
        }

    # === Retention ===

    async def _enforce_limits(self) -> None:
        """Enforce history limits."""
        # Remove old records
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        old_ids = [
            rid for rid, record in self._history.items()
            if record.started_at and record.started_at < cutoff
        ]
        for rid in old_ids:
            await self.delete(rid)

        # Remove excess records
        if len(self._history) > self.max_history_count:
            # Sort by started_at and remove oldest
            records = sorted(
                self._history.values(),
                key=lambda r: r.started_at or datetime.min,
            )
            excess = len(records) - self.max_history_count
            for record in records[:excess]:
                await self.delete(record.id)

    async def cleanup(
        self,
        older_than_days: Optional[int] = None,
    ) -> int:
        """Clean up old history records."""
        days = older_than_days or self.retention_days
        cutoff = datetime.now() - timedelta(days=days)

        deleted = 0
        async with self._lock:
            old_ids = [
                rid for rid, record in self._history.items()
                if record.started_at and record.started_at < cutoff
            ]
            for rid in old_ids:
                if rid in self._history:
                    del self._history[rid]
                    deleted += 1

        logger.info("history_cleaned", deleted=deleted)
        return deleted

    # === Persistence ===

    async def _load_history(self) -> None:
        """Load history from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            history_file = self.persistence_path / "execution_history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    data = json.load(f)

                for record_data in data.get("records", []):
                    record = ExecutionRecord.from_dict(record_data)
                    self._history[record.id] = record

                    # Update indices
                    self._by_workflow[record.workflow_id].append(record.id)
                    self._by_status[record.status].append(record.id)
                    if record.started_at:
                        date_key = record.started_at.strftime("%Y-%m-%d")
                        self._by_date[date_key].append(record.id)

        except Exception as e:
            logger.error("load_history_error", error=str(e))

    async def _save_history(self) -> None:
        """Save history to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            history_file = self.persistence_path / "execution_history.json"

            data = {
                "records": [r.to_dict() for r in self._history.values()],
                "saved_at": datetime.now().isoformat(),
            }

            with open(history_file, "w") as f:
                json.dump(data, f, default=str)

        except Exception as e:
            logger.error("save_history_error", error=str(e))


from dataclasses import dataclass, field


@dataclass
class ExecutionRecord:
    """A historical execution record."""
    id: str
    workflow_id: str
    workflow_name: str = ""
    workflow_version: int = 1

    # Trigger
    trigger_type: Optional[str] = None

    # Status
    status: ExecutionStatus = ExecutionStatus.COMPLETED

    # Data
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # Metrics
    step_count: int = 0
    duration_ms: float = 0.0

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Owner
    initiated_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "trigger_type": self.trigger_type,
            "status": self.status.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "step_count": self.step_count,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "initiated_by": self.initiated_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRecord":
        """Create from dictionary."""
        record = cls(
            id=data["id"],
            workflow_id=data["workflow_id"],
            workflow_name=data.get("workflow_name", ""),
            workflow_version=data.get("workflow_version", 1),
            trigger_type=data.get("trigger_type"),
            status=ExecutionStatus(data.get("status", "completed")),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            error=data.get("error"),
            step_count=data.get("step_count", 0),
            duration_ms=data.get("duration_ms", 0.0),
            initiated_by=data.get("initiated_by"),
        )

        if data.get("started_at"):
            record.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            record.completed_at = datetime.fromisoformat(data["completed_at"])

        return record
