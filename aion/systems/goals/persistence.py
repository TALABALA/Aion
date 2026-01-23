"""
AION Goal Persistence Layer

Durable storage for goals and related data:
- File-based persistence
- Goal serialization/deserialization
- Event history storage
- Backup and recovery
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog

from aion.systems.goals.types import Goal, GoalEvent, Objective

logger = structlog.get_logger(__name__)


class GoalPersistence:
    """
    Persistent storage for goals.

    Features:
    - JSON file-based storage
    - Atomic writes with backup
    - Event log rotation
    - Automatic recovery
    """

    def __init__(
        self,
        data_dir: str = "data/goals",
        goals_file: str = "goals.json",
        objectives_file: str = "objectives.json",
        events_file: str = "events.json",
        max_events: int = 10000,
        backup_enabled: bool = True,
    ):
        self._data_dir = Path(data_dir)
        self._goals_file = self._data_dir / goals_file
        self._objectives_file = self._data_dir / objectives_file
        self._events_file = self._data_dir / events_file

        self._max_events = max_events
        self._backup_enabled = backup_enabled

        # Lock for file operations
        self._lock = asyncio.Lock()

        # In-memory cache for events (to batch writes)
        self._event_buffer: list[GoalEvent] = []
        self._event_buffer_size = 100

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize persistence layer."""
        if self._initialized:
            return

        logger.info("Initializing Goal Persistence", data_dir=str(self._data_dir))

        # Create data directory
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize files if they don't exist
        for filepath in [self._goals_file, self._objectives_file, self._events_file]:
            if not filepath.exists():
                await self._write_json(filepath, [])

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown and flush pending writes."""
        # Flush event buffer
        if self._event_buffer:
            await self._flush_events()

        self._initialized = False

    # === Goals ===

    async def save_goal(self, goal: Goal) -> None:
        """Save a goal."""
        async with self._lock:
            goals = await self._read_goals()

            # Update or add
            found = False
            for i, g in enumerate(goals):
                if g.id == goal.id:
                    goals[i] = goal
                    found = True
                    break

            if not found:
                goals.append(goal)

            await self._write_goals(goals)

    async def load_goal(self, goal_id: str) -> Optional[Goal]:
        """Load a goal by ID."""
        goals = await self._read_goals()
        for goal in goals:
            if goal.id == goal_id:
                return goal
        return None

    async def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal."""
        async with self._lock:
            goals = await self._read_goals()

            original_len = len(goals)
            goals = [g for g in goals if g.id != goal_id]

            if len(goals) < original_len:
                await self._write_goals(goals)
                return True

            return False

    async def load_all_goals(self) -> list[Goal]:
        """Load all goals."""
        return await self._read_goals()

    async def _read_goals(self) -> list[Goal]:
        """Read goals from file."""
        try:
            data = await self._read_json(self._goals_file)
            return [Goal.from_dict(g) for g in data]
        except Exception as e:
            logger.error(f"Error reading goals: {e}")
            return []

    async def _write_goals(self, goals: list[Goal]) -> None:
        """Write goals to file."""
        data = [g.to_dict() for g in goals]
        await self._write_json(self._goals_file, data)

    # === Objectives ===

    async def save_objective(self, objective: Objective) -> None:
        """Save an objective."""
        async with self._lock:
            objectives = await self._read_objectives()

            # Update or add
            found = False
            for i, o in enumerate(objectives):
                if o.id == objective.id:
                    objectives[i] = objective
                    found = True
                    break

            if not found:
                objectives.append(objective)

            await self._write_objectives(objectives)

    async def load_objective(self, objective_id: str) -> Optional[Objective]:
        """Load an objective by ID."""
        objectives = await self._read_objectives()
        for obj in objectives:
            if obj.id == objective_id:
                return obj
        return None

    async def delete_objective(self, objective_id: str) -> bool:
        """Delete an objective."""
        async with self._lock:
            objectives = await self._read_objectives()

            original_len = len(objectives)
            objectives = [o for o in objectives if o.id != objective_id]

            if len(objectives) < original_len:
                await self._write_objectives(objectives)
                return True

            return False

    async def load_all_objectives(self) -> list[Objective]:
        """Load all objectives."""
        return await self._read_objectives()

    async def _read_objectives(self) -> list[Objective]:
        """Read objectives from file."""
        try:
            data = await self._read_json(self._objectives_file)
            return [Objective.from_dict(o) for o in data]
        except Exception as e:
            logger.error(f"Error reading objectives: {e}")
            return []

    async def _write_objectives(self, objectives: list[Objective]) -> None:
        """Write objectives to file."""
        data = [o.to_dict() for o in objectives]
        await self._write_json(self._objectives_file, data)

    # === Events ===

    async def save_event(self, event: GoalEvent) -> None:
        """Save an event (buffered)."""
        self._event_buffer.append(event)

        # Flush if buffer is full
        if len(self._event_buffer) >= self._event_buffer_size:
            await self._flush_events()

    async def load_recent_events(self, limit: int = 1000) -> list[GoalEvent]:
        """Load recent events."""
        # Flush buffer first
        if self._event_buffer:
            await self._flush_events()

        try:
            data = await self._read_json(self._events_file)
            events = [GoalEvent.from_dict(e) for e in data]
            return events[-limit:]
        except Exception as e:
            logger.error(f"Error reading events: {e}")
            return []

    async def load_events_for_goal(
        self, goal_id: str, limit: int = 100
    ) -> list[GoalEvent]:
        """Load events for a specific goal."""
        all_events = await self.load_recent_events(limit=self._max_events)
        goal_events = [e for e in all_events if e.goal_id == goal_id]
        return goal_events[-limit:]

    async def _flush_events(self) -> None:
        """Flush event buffer to disk."""
        if not self._event_buffer:
            return

        async with self._lock:
            try:
                # Read existing events
                data = await self._read_json(self._events_file)

                # Add new events
                for event in self._event_buffer:
                    data.append(event.to_dict())

                # Truncate if too many
                if len(data) > self._max_events:
                    data = data[-self._max_events :]

                # Write back
                await self._write_json(self._events_file, data)

                # Clear buffer
                self._event_buffer.clear()

            except Exception as e:
                logger.error(f"Error flushing events: {e}")

    # === File Operations ===

    async def _read_json(self, filepath: Path) -> list:
        """Read JSON file."""
        try:
            if not filepath.exists():
                return []

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None, lambda: filepath.read_text(encoding="utf-8")
            )

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filepath}: {e}")
            # Try to recover from backup
            backup = filepath.with_suffix(".json.bak")
            if backup.exists():
                content = backup.read_text(encoding="utf-8")
                return json.loads(content)
            return []

        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return []

    async def _write_json(self, filepath: Path, data: list) -> None:
        """Write JSON file with atomic operation."""
        try:
            content = json.dumps(data, indent=2, default=str)

            # Create backup if enabled
            if self._backup_enabled and filepath.exists():
                backup = filepath.with_suffix(".json.bak")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: backup.write_text(
                        filepath.read_text(encoding="utf-8"),
                        encoding="utf-8"
                    )
                )

            # Write to temp file first
            temp_file = filepath.with_suffix(".json.tmp")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: temp_file.write_text(content, encoding="utf-8")
            )

            # Atomic rename
            await loop.run_in_executor(None, lambda: temp_file.replace(filepath))

        except Exception as e:
            logger.error(f"Error writing {filepath}: {e}")
            raise

    # === Backup and Recovery ===

    async def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a full backup."""
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        backup_dir = self._data_dir / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy all data files
        for filepath in [self._goals_file, self._objectives_file, self._events_file]:
            if filepath.exists():
                backup_path = backup_dir / filepath.name
                content = filepath.read_text(encoding="utf-8")
                backup_path.write_text(content, encoding="utf-8")

        logger.info(f"Created backup: {backup_name}")
        return str(backup_dir)

    async def restore_backup(self, backup_name: str) -> bool:
        """Restore from a backup."""
        backup_dir = self._data_dir / "backups" / backup_name

        if not backup_dir.exists():
            logger.error(f"Backup not found: {backup_name}")
            return False

        async with self._lock:
            # Restore all files
            for filepath in [self._goals_file, self._objectives_file, self._events_file]:
                backup_path = backup_dir / filepath.name
                if backup_path.exists():
                    content = backup_path.read_text(encoding="utf-8")
                    filepath.write_text(content, encoding="utf-8")

        logger.info(f"Restored backup: {backup_name}")
        return True

    async def list_backups(self) -> list[str]:
        """List available backups."""
        backup_dir = self._data_dir / "backups"
        if not backup_dir.exists():
            return []

        backups = []
        for item in backup_dir.iterdir():
            if item.is_dir():
                backups.append(item.name)

        return sorted(backups, reverse=True)

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get persistence statistics."""
        stats = {
            "data_dir": str(self._data_dir),
            "event_buffer_size": len(self._event_buffer),
            "files": {},
        }

        for name, filepath in [
            ("goals", self._goals_file),
            ("objectives", self._objectives_file),
            ("events", self._events_file),
        ]:
            if filepath.exists():
                stats["files"][name] = {
                    "path": str(filepath),
                    "size_bytes": filepath.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        filepath.stat().st_mtime
                    ).isoformat(),
                }

        return stats

    async def compact_events(self) -> int:
        """Compact event log by removing old events."""
        async with self._lock:
            data = await self._read_json(self._events_file)

            original_count = len(data)

            if len(data) > self._max_events:
                data = data[-self._max_events :]
                await self._write_json(self._events_file, data)

            removed = original_count - len(data)

            if removed > 0:
                logger.info(f"Compacted events: removed {removed}")

            return removed
