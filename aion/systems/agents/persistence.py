"""
AION Multi-Agent Persistence

State persistence and recovery for multi-agent system.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog

from aion.systems.agents.types import (
    AgentInstance,
    AgentProfile,
    Team,
    TeamTask,
    Message,
)

logger = structlog.get_logger(__name__)


class MultiAgentPersistence:
    """
    Persistence layer for multi-agent system state.

    Features:
    - Agent state snapshots
    - Team state persistence
    - Message history storage
    - Task state recovery
    """

    def __init__(
        self,
        storage_dir: str = "data/agents",
        auto_save: bool = True,
        save_interval_seconds: int = 300,
    ):
        self.storage_dir = Path(storage_dir)
        self.auto_save = auto_save
        self.save_interval_seconds = save_interval_seconds

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._last_save: Optional[datetime] = None

    # === Agent Persistence ===

    def save_agent(self, agent: AgentInstance) -> bool:
        """Save agent state to disk."""
        try:
            agent_dir = self.storage_dir / "agents"
            agent_dir.mkdir(exist_ok=True)

            filepath = agent_dir / f"{agent.id}.json"

            data = {
                "instance": agent.to_dict(),
                "saved_at": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug("Saved agent", agent_id=agent.id[:8])
            return True

        except Exception as e:
            logger.error("Failed to save agent", agent_id=agent.id[:8], error=str(e))
            return False

    def load_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Load agent state from disk."""
        try:
            filepath = self.storage_dir / "agents" / f"{agent_id}.json"

            if not filepath.exists():
                return None

            with open(filepath) as f:
                data = json.load(f)

            return AgentInstance.from_dict(data["instance"])

        except Exception as e:
            logger.error("Failed to load agent", agent_id=agent_id[:8], error=str(e))
            return None

    def list_saved_agents(self) -> list[str]:
        """List IDs of saved agents."""
        agent_dir = self.storage_dir / "agents"
        if not agent_dir.exists():
            return []

        return [
            f.stem for f in agent_dir.glob("*.json")
        ]

    def delete_agent(self, agent_id: str) -> bool:
        """Delete saved agent state."""
        try:
            filepath = self.storage_dir / "agents" / f"{agent_id}.json"
            if filepath.exists():
                filepath.unlink()
            return True
        except Exception as e:
            logger.error("Failed to delete agent", agent_id=agent_id[:8], error=str(e))
            return False

    # === Team Persistence ===

    def save_team(self, team: Team) -> bool:
        """Save team state to disk."""
        try:
            team_dir = self.storage_dir / "teams"
            team_dir.mkdir(exist_ok=True)

            filepath = team_dir / f"{team.id}.json"

            data = {
                "team": team.to_dict(),
                "saved_at": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug("Saved team", team_id=team.id[:8])
            return True

        except Exception as e:
            logger.error("Failed to save team", team_id=team.id[:8], error=str(e))
            return False

    def load_team(self, team_id: str) -> Optional[Team]:
        """Load team state from disk."""
        try:
            filepath = self.storage_dir / "teams" / f"{team_id}.json"

            if not filepath.exists():
                return None

            with open(filepath) as f:
                data = json.load(f)

            return Team.from_dict(data["team"])

        except Exception as e:
            logger.error("Failed to load team", team_id=team_id[:8], error=str(e))
            return None

    def list_saved_teams(self) -> list[str]:
        """List IDs of saved teams."""
        team_dir = self.storage_dir / "teams"
        if not team_dir.exists():
            return []

        return [f.stem for f in team_dir.glob("*.json")]

    def delete_team(self, team_id: str) -> bool:
        """Delete saved team state."""
        try:
            filepath = self.storage_dir / "teams" / f"{team_id}.json"
            if filepath.exists():
                filepath.unlink()
            return True
        except Exception as e:
            logger.error("Failed to delete team", team_id=team_id[:8], error=str(e))
            return False

    # === Task Persistence ===

    def save_task(self, task: TeamTask) -> bool:
        """Save task state to disk."""
        try:
            task_dir = self.storage_dir / "tasks"
            task_dir.mkdir(exist_ok=True)

            filepath = task_dir / f"{task.id}.json"

            data = {
                "task": task.to_dict(),
                "saved_at": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug("Saved task", task_id=task.id[:8])
            return True

        except Exception as e:
            logger.error("Failed to save task", task_id=task.id[:8], error=str(e))
            return False

    def load_task(self, task_id: str) -> Optional[TeamTask]:
        """Load task state from disk."""
        try:
            filepath = self.storage_dir / "tasks" / f"{task_id}.json"

            if not filepath.exists():
                return None

            with open(filepath) as f:
                data = json.load(f)

            return TeamTask.from_dict(data["task"])

        except Exception as e:
            logger.error("Failed to load task", task_id=task_id[:8], error=str(e))
            return None

    def list_saved_tasks(self) -> list[str]:
        """List IDs of saved tasks."""
        task_dir = self.storage_dir / "tasks"
        if not task_dir.exists():
            return []

        return [f.stem for f in task_dir.glob("*.json")]

    # === Message History ===

    def save_message_history(
        self,
        messages: list[Message],
        context_id: str,
    ) -> bool:
        """Save message history."""
        try:
            msg_dir = self.storage_dir / "messages"
            msg_dir.mkdir(exist_ok=True)

            filepath = msg_dir / f"{context_id}.json"

            data = {
                "messages": [m.to_dict() for m in messages],
                "saved_at": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug("Saved message history", context_id=context_id[:8], count=len(messages))
            return True

        except Exception as e:
            logger.error("Failed to save messages", context_id=context_id[:8], error=str(e))
            return False

    def load_message_history(self, context_id: str) -> list[Message]:
        """Load message history."""
        try:
            filepath = self.storage_dir / "messages" / f"{context_id}.json"

            if not filepath.exists():
                return []

            with open(filepath) as f:
                data = json.load(f)

            return [Message.from_dict(m) for m in data["messages"]]

        except Exception as e:
            logger.error("Failed to load messages", context_id=context_id[:8], error=str(e))
            return []

    # === Bulk Operations ===

    def save_snapshot(
        self,
        agents: list[AgentInstance],
        teams: list[Team],
        tasks: list[TeamTask],
        snapshot_id: Optional[str] = None,
    ) -> str:
        """Save a complete system snapshot."""
        snapshot_id = snapshot_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        snapshot_dir = self.storage_dir / "snapshots" / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save all components
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().isoformat(),
            "agents": [a.to_dict() for a in agents],
            "teams": [t.to_dict() for t in teams],
            "tasks": [t.to_dict() for t in tasks],
        }

        with open(snapshot_dir / "snapshot.json", "w") as f:
            json.dump(snapshot_data, f, indent=2, default=str)

        logger.info(
            "Created snapshot",
            snapshot_id=snapshot_id,
            agents=len(agents),
            teams=len(teams),
            tasks=len(tasks),
        )

        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> Optional[dict[str, Any]]:
        """Load a system snapshot."""
        try:
            filepath = self.storage_dir / "snapshots" / snapshot_id / "snapshot.json"

            if not filepath.exists():
                return None

            with open(filepath) as f:
                data = json.load(f)

            # Convert back to objects
            return {
                "snapshot_id": data["snapshot_id"],
                "created_at": data["created_at"],
                "agents": [AgentInstance.from_dict(a) for a in data["agents"]],
                "teams": [Team.from_dict(t) for t in data["teams"]],
                "tasks": [TeamTask.from_dict(t) for t in data["tasks"]],
            }

        except Exception as e:
            logger.error("Failed to load snapshot", snapshot_id=snapshot_id, error=str(e))
            return None

    def list_snapshots(self) -> list[str]:
        """List available snapshots."""
        snapshot_dir = self.storage_dir / "snapshots"
        if not snapshot_dir.exists():
            return []

        return [
            d.name for d in snapshot_dir.iterdir()
            if d.is_dir() and (d / "snapshot.json").exists()
        ]

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        try:
            import shutil
            snapshot_dir = self.storage_dir / "snapshots" / snapshot_id
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            return True
        except Exception as e:
            logger.error("Failed to delete snapshot", snapshot_id=snapshot_id, error=str(e))
            return False

    # === Cleanup ===

    def cleanup_old_data(self, max_age_days: int = 30) -> int:
        """Clean up data older than specified days."""
        cleaned = 0
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)

        for subdir in ["agents", "teams", "tasks", "messages"]:
            dir_path = self.storage_dir / subdir
            if not dir_path.exists():
                continue

            for filepath in dir_path.glob("*.json"):
                if filepath.stat().st_mtime < cutoff:
                    filepath.unlink()
                    cleaned += 1

        logger.info("Cleaned up old data", removed=cleaned, max_age_days=max_age_days)
        return cleaned

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "storage_dir": str(self.storage_dir),
            "agents_saved": len(self.list_saved_agents()),
            "teams_saved": len(self.list_saved_teams()),
            "tasks_saved": len(self.list_saved_tasks()),
            "snapshots": len(self.list_snapshots()),
        }

        # Calculate total size
        total_size = 0
        for filepath in self.storage_dir.rglob("*.json"):
            total_size += filepath.stat().st_size

        stats["total_size_bytes"] = total_size
        stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        return stats
