"""AION Branch Manager - Timeline branching for what-if analysis.

Provides:
- BranchManager: DAG-based branch management for creating, switching,
  merging, and comparing timeline branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Branch:
    """A timeline branch."""

    name: str
    snapshot_ids: List[str] = field(default_factory=list)
    parent_branch: Optional[str] = None
    fork_snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class BranchManager:
    """DAG-based branch management for timeline branching.

    Features:
    - Create branches from any snapshot point.
    - Switch between branches.
    - Branch hierarchy (parent/child tracking).
    - Branch comparison.
    - Branch metadata for annotation.
    """

    def __init__(self) -> None:
        self._branches: Dict[str, Branch] = {
            "main": Branch(name="main"),
        }
        self._current: str = "main"

    @property
    def current_branch(self) -> str:
        return self._current

    @property
    def current(self) -> Branch:
        return self._branches[self._current]

    def create(
        self,
        name: str,
        from_snapshot_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Branch:
        """Create a new branch."""
        if name in self._branches:
            raise ValueError(f"Branch already exists: {name}")

        branch = Branch(
            name=name,
            parent_branch=self._current,
            fork_snapshot_id=from_snapshot_id,
            metadata=metadata or {},
        )

        if from_snapshot_id:
            branch.snapshot_ids.append(from_snapshot_id)

        self._branches[name] = branch
        logger.info("branch_created", name=name, parent=self._current)
        return branch

    def switch(self, name: str) -> Branch:
        """Switch to a branch."""
        if name not in self._branches:
            raise ValueError(f"Branch not found: {name}")
        self._current = name
        logger.info("branch_switched", name=name)
        return self._branches[name]

    def add_snapshot(self, snapshot_id: str, branch: Optional[str] = None) -> None:
        """Add a snapshot to a branch."""
        branch_name = branch or self._current
        b = self._branches.get(branch_name)
        if b is None:
            raise ValueError(f"Branch not found: {branch_name}")
        b.snapshot_ids.append(snapshot_id)

    def get(self, name: str) -> Optional[Branch]:
        return self._branches.get(name)

    def list_branches(self) -> List[str]:
        return list(self._branches.keys())

    def list_active(self) -> List[str]:
        return [n for n, b in self._branches.items() if b.is_active]

    def get_children(self, name: str) -> List[str]:
        """Get child branches."""
        return [
            n for n, b in self._branches.items()
            if b.parent_branch == name
        ]

    def get_lineage(self, name: str) -> List[str]:
        """Get branch lineage from root to given branch."""
        lineage: List[str] = []
        current = name
        visited: Set[str] = set()
        while current and current not in visited:
            visited.add(current)
            lineage.append(current)
            branch = self._branches.get(current)
            current = branch.parent_branch if branch else None
        lineage.reverse()
        return lineage

    def get_snapshots(self, name: Optional[str] = None) -> List[str]:
        branch_name = name or self._current
        b = self._branches.get(branch_name)
        return list(b.snapshot_ids) if b else []

    def latest_snapshot(self, name: Optional[str] = None) -> Optional[str]:
        snapshots = self.get_snapshots(name)
        return snapshots[-1] if snapshots else None

    def delete(self, name: str) -> bool:
        """Soft-delete a branch (marks inactive)."""
        if name == "main":
            logger.warning("cannot_delete_main_branch")
            return False
        b = self._branches.get(name)
        if b is None:
            return False
        b.is_active = False
        if self._current == name:
            self._current = b.parent_branch or "main"
        logger.info("branch_deleted", name=name)
        return True

    def annotate(self, name: str, key: str, value: Any) -> None:
        """Add metadata to a branch."""
        b = self._branches.get(name)
        if b:
            b.metadata[key] = value

    @property
    def branch_count(self) -> int:
        return len(self._branches)

    def tree_summary(self) -> Dict[str, Any]:
        """Get a summary of the branch tree."""
        return {
            name: {
                "parent": b.parent_branch,
                "snapshots": len(b.snapshot_ids),
                "active": b.is_active,
                "children": self.get_children(name),
            }
            for name, b in self._branches.items()
        }
