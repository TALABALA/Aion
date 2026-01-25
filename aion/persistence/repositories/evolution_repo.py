"""
AION Evolution Repository

Persistence for the self-improvement engine including:
- Evolution checkpoints
- Hypotheses history
- Performance snapshots
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

# Import evolution types
try:
    from aion.systems.evolution.engine import (
        EvolutionCheckpoint,
        PerformanceSnapshot,
    )
    from aion.systems.evolution.hypothesis import (
        Hypothesis,
        HypothesisStatus,
        HypothesisType,
    )
except ImportError:
    # Fallback types
    from enum import Enum
    from dataclasses import dataclass, field

    class HypothesisStatus(str, Enum):
        PENDING = "pending"
        TESTING = "testing"
        VALIDATED = "validated"
        REJECTED = "rejected"
        APPLIED = "applied"

    class HypothesisType(str, Enum):
        PARAMETER_ADJUSTMENT = "parameter_adjustment"
        THRESHOLD_TUNING = "threshold_tuning"
        ALGORITHM_SWITCH = "algorithm_switch"

    @dataclass
    class EvolutionCheckpoint:
        id: str
        timestamp: datetime
        parameters: dict
        performance: dict
        applied_hypotheses: list

    @dataclass
    class PerformanceSnapshot:
        timestamp: datetime
        metrics: dict
        parameters: dict
        metadata: dict = field(default_factory=dict)

    @dataclass
    class Hypothesis:
        id: str
        name: str
        hypothesis_type: HypothesisType
        target: str
        current_value: Any
        proposed_value: Any
        rationale: str
        status: HypothesisStatus = HypothesisStatus.PENDING
        confidence: float = 0.0
        expected_improvement: float = 0.0
        result_improvement: Optional[float] = None
        created_at: datetime = field(default_factory=datetime.now)
        tested_at: Optional[datetime] = None
        decided_at: Optional[datetime] = None


class EvolutionRepository(BaseRepository[EvolutionCheckpoint]):
    """
    Repository for evolution checkpoint persistence.

    Features:
    - Full checkpoint serialization
    - Performance history tracking
    - Hypothesis management
    """

    def __init__(
        self,
        db: DatabaseManager,
        cache: Optional[CacheManager] = None,
    ):
        super().__init__(db, cache)
        self._table_name = "evolution_checkpoints"
        self._soft_delete_column = None

    def _serialize(self, checkpoint: EvolutionCheckpoint) -> dict[str, Any]:
        """Serialize EvolutionCheckpoint to database row."""
        return {
            "id": checkpoint.id,
            "parameters": self._to_json(checkpoint.parameters),
            "parameter_bounds": self._to_json({}),  # Not stored in this model
            "performance_metrics": self._to_json(checkpoint.performance),
            "applied_hypotheses": self._to_json(checkpoint.applied_hypotheses),
            "created_at": self._from_datetime(checkpoint.timestamp),
        }

    def _deserialize(self, row: dict[str, Any]) -> EvolutionCheckpoint:
        """Deserialize database row to EvolutionCheckpoint."""
        return EvolutionCheckpoint(
            id=row["id"],
            timestamp=self._to_datetime(row.get("created_at")) or datetime.now(),
            parameters=self._from_json(row.get("parameters")) or {},
            performance=self._from_json(row.get("performance_metrics")) or {},
            applied_hypotheses=self._from_json(row.get("applied_hypotheses")) or [],
        )

    async def get_latest(self) -> Optional[EvolutionCheckpoint]:
        """Get the most recent checkpoint."""
        return await self.find_one_where(
            "1=1",
            (),
            QueryOptions(order_by="created_at DESC", limit=1),
        )

    async def get_recent(
        self,
        limit: int = 10,
    ) -> list[EvolutionCheckpoint]:
        """Get recent checkpoints."""
        return await self.get_all(
            QueryOptions(order_by="created_at DESC", limit=limit)
        )

    # === Hypothesis Operations ===

    async def save_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Save a hypothesis."""
        hypothesis_id = hypothesis.id or str(uuid.uuid4())

        query = """
            INSERT INTO evolution_hypotheses (
                id, parameter, current_value, proposed_value, rationale,
                status, confidence, expected_improvement, actual_improvement,
                created_at, tested_at, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                confidence = excluded.confidence,
                actual_improvement = excluded.actual_improvement,
                tested_at = excluded.tested_at,
                decided_at = excluded.decided_at
        """

        await self.db.execute(query, (
            hypothesis_id,
            hypothesis.target,
            self._to_json(hypothesis.current_value),
            self._to_json(hypothesis.proposed_value),
            hypothesis.rationale,
            hypothesis.status.value if hasattr(hypothesis.status, 'value') else str(hypothesis.status),
            hypothesis.confidence,
            hypothesis.expected_improvement,
            hypothesis.result_improvement,
            self._from_datetime(hypothesis.created_at),
            self._from_datetime(hypothesis.tested_at),
            self._from_datetime(hypothesis.decided_at),
        ))

        return hypothesis_id

    async def get_hypothesis(self, id: str) -> Optional[Hypothesis]:
        """Get a hypothesis by ID."""
        query = "SELECT * FROM evolution_hypotheses WHERE id = ?"
        row = await self.db.fetch_one(query, (id,))

        if not row:
            return None

        return self._deserialize_hypothesis(row)

    def _deserialize_hypothesis(self, row: dict[str, Any]) -> Hypothesis:
        """Deserialize a hypothesis row."""
        return Hypothesis(
            id=row["id"],
            name=row.get("parameter", ""),
            hypothesis_type=HypothesisType.PARAMETER_ADJUSTMENT,
            target=row.get("parameter", ""),
            current_value=self._from_json(row.get("current_value")),
            proposed_value=self._from_json(row.get("proposed_value")),
            rationale=row.get("rationale", ""),
            status=HypothesisStatus(row.get("status", "pending")),
            confidence=row.get("confidence", 0.0),
            expected_improvement=row.get("expected_improvement", 0.0),
            result_improvement=row.get("actual_improvement"),
            created_at=self._to_datetime(row.get("created_at")) or datetime.now(),
            tested_at=self._to_datetime(row.get("tested_at")),
            decided_at=self._to_datetime(row.get("decided_at")),
        )

    async def find_hypotheses_by_status(
        self,
        status: HypothesisStatus,
        limit: int = 100,
    ) -> list[Hypothesis]:
        """Find hypotheses by status."""
        status_value = status.value if hasattr(status, 'value') else str(status)

        query = """
            SELECT * FROM evolution_hypotheses
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
        """

        rows = await self.db.fetch_all(query, (status_value, limit))
        return [self._deserialize_hypothesis(row) for row in rows]

    async def find_pending_hypotheses(self) -> list[Hypothesis]:
        """Find pending hypotheses."""
        return await self.find_hypotheses_by_status(HypothesisStatus.PENDING)

    async def find_applied_hypotheses(
        self,
        limit: int = 100,
    ) -> list[Hypothesis]:
        """Find applied hypotheses."""
        return await self.find_hypotheses_by_status(HypothesisStatus.APPLIED, limit)

    async def update_hypothesis_status(
        self,
        id: str,
        status: HypothesisStatus,
        improvement: Optional[float] = None,
    ) -> bool:
        """Update hypothesis status."""
        status_value = status.value if hasattr(status, 'value') else str(status)

        fields = {
            "status": status_value,
            "decided_at": datetime.now().isoformat(),
        }

        if improvement is not None:
            fields["actual_improvement"] = improvement

        if status == HypothesisStatus.TESTING:
            fields["tested_at"] = datetime.now().isoformat()

        set_clause = ", ".join([f"{k} = ?" for k in fields.keys()])
        query = f"UPDATE evolution_hypotheses SET {set_clause} WHERE id = ?"

        await self.db.execute(query, (*fields.values(), id))
        return True

    # === Performance Snapshot Operations ===

    async def save_performance_snapshot(
        self,
        snapshot: PerformanceSnapshot,
    ) -> int:
        """Save a performance snapshot."""
        query = """
            INSERT INTO performance_snapshots (metrics, parameters, metadata, timestamp)
            VALUES (?, ?, ?, ?)
        """

        result = await self.db.execute(query, (
            self._to_json(snapshot.metrics),
            self._to_json(snapshot.parameters),
            self._to_json(snapshot.metadata),
            snapshot.timestamp.isoformat(),
        ))

        return result

    async def get_performance_history(
        self,
        hours: int = 24,
        limit: int = 1000,
    ) -> list[PerformanceSnapshot]:
        """Get recent performance history."""
        query = """
            SELECT * FROM performance_snapshots
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
            LIMIT ?
        """

        rows = await self.db.fetch_all(query, (hours, limit))

        return [
            PerformanceSnapshot(
                timestamp=self._to_datetime(row.get("timestamp")) or datetime.now(),
                metrics=self._from_json(row.get("metrics")) or {},
                parameters=self._from_json(row.get("parameters")) or {},
                metadata=self._from_json(row.get("metadata")) or {},
            )
            for row in rows
        ]

    async def get_performance_summary(
        self,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Get performance summary for a time period."""
        snapshots = await self.get_performance_history(hours=hours)

        if not snapshots:
            return {
                "period_hours": hours,
                "sample_count": 0,
                "metrics": {},
            }

        # Aggregate metrics
        all_metrics: dict[str, list[float]] = {}
        for snapshot in snapshots:
            for metric, value in snapshot.metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

        # Calculate statistics
        import statistics

        metric_stats = {}
        for metric, values in all_metrics.items():
            if values:
                metric_stats[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                }

        return {
            "period_hours": hours,
            "sample_count": len(snapshots),
            "metrics": metric_stats,
            "first_timestamp": self._from_datetime(snapshots[-1].timestamp) if snapshots else None,
            "last_timestamp": self._from_datetime(snapshots[0].timestamp) if snapshots else None,
        }

    async def cleanup_old_data(
        self,
        checkpoint_days: int = 30,
        hypothesis_days: int = 90,
        snapshot_days: int = 7,
    ) -> dict[str, int]:
        """Clean up old evolution data."""
        results = {}

        # Clean old checkpoints
        query = """
            DELETE FROM evolution_checkpoints
            WHERE created_at < datetime('now', '-' || ? || ' days')
        """
        await self.db.execute(query, (checkpoint_days,))
        results["checkpoints_deleted"] = 0

        # Clean old hypotheses (except applied ones)
        query = """
            DELETE FROM evolution_hypotheses
            WHERE created_at < datetime('now', '-' || ? || ' days')
            AND status != 'applied'
        """
        await self.db.execute(query, (hypothesis_days,))
        results["hypotheses_deleted"] = 0

        # Clean old performance snapshots
        query = """
            DELETE FROM performance_snapshots
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """
        await self.db.execute(query, (snapshot_days,))
        results["snapshots_deleted"] = 0

        return results

    async def get_statistics(self) -> dict[str, Any]:
        """Get evolution system statistics."""
        checkpoint_query = "SELECT COUNT(*) as count FROM evolution_checkpoints"
        hypothesis_query = """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                COUNT(CASE WHEN status = 'applied' THEN 1 END) as applied,
                COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected,
                AVG(CASE WHEN status = 'applied' THEN actual_improvement END) as avg_improvement
            FROM evolution_hypotheses
        """
        snapshot_query = "SELECT COUNT(*) as count FROM performance_snapshots"

        checkpoints = await self.db.fetch_one(checkpoint_query)
        hypotheses = await self.db.fetch_one(hypothesis_query)
        snapshots = await self.db.fetch_one(snapshot_query)

        return {
            "checkpoints": checkpoints["count"] if checkpoints else 0,
            "hypotheses": {
                "total": hypotheses["total"] if hypotheses else 0,
                "pending": hypotheses["pending"] if hypotheses else 0,
                "applied": hypotheses["applied"] if hypotheses else 0,
                "rejected": hypotheses["rejected"] if hypotheses else 0,
                "avg_improvement": hypotheses["avg_improvement"] if hypotheses else 0,
            },
            "performance_snapshots": snapshots["count"] if snapshots else 0,
        }
