"""AION Timeline Manager - Orchestrates snapshots, branching, and replay.

Provides:
- TimelineManager: High-level API for time-travel, branching, comparison,
  and deterministic replay of simulation timelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import structlog

from aion.simulation.timeline.branching import BranchManager
from aion.simulation.timeline.replay import ReplayEngine
from aion.simulation.timeline.snapshots import SnapshotStore
from aion.simulation.types import (
    EventType,
    SimulationEvent,
    TimelineSnapshot,
    WorldState,
)

if TYPE_CHECKING:
    from aion.simulation.world.engine import WorldEngine

logger = structlog.get_logger(__name__)


class TimelineManager:
    """Manages simulation timelines with branching and replay.

    SOTA features:
    - Content-addressable snapshot storage with deduplication.
    - DAG-based branch management for what-if analysis.
    - Deterministic replay engine with selective filtering.
    - Snapshot comparison and diff analysis.
    - Time-travel (rewind/fast-forward).
    """

    def __init__(self, world_engine: "WorldEngine") -> None:
        self.world_engine = world_engine

        # Sub-components
        self.snapshots = SnapshotStore()
        self.branches = BranchManager()
        self.replay = ReplayEngine(world_engine)

        # Hook replay recording into world engine
        world_engine.add_post_tick_hook(self._on_tick)

    def _on_tick(self, state: WorldState, events: List[SimulationEvent]) -> None:
        """Post-tick hook: record events for replay if recording."""
        for event in events:
            self.replay.record_event(event)

    # -- Snapshot Operations --

    def create_snapshot(
        self,
        description: str = "",
        tags: Optional[set] = None,
    ) -> TimelineSnapshot:
        """Create a snapshot of the current world state."""
        state = self.world_engine.clone_state()

        # Find parent snapshot on current branch
        parent_id = self.branches.latest_snapshot()

        snapshot = TimelineSnapshot(
            tick=state.tick,
            simulation_time=state.simulation_time,
            world_state=state,
            branch_name=self.branches.current_branch,
            parent_snapshot_id=parent_id,
            description=description,
            tags=tags or set(),
        )

        stored_id = self.snapshots.store(snapshot)

        # If deduplicated, update reference to existing snapshot
        if stored_id != snapshot.id:
            snapshot = self.snapshots.get(stored_id)
        else:
            self.branches.add_snapshot(snapshot.id)

        logger.debug(
            "snapshot_created",
            tick=state.tick,
            snapshot_id=snapshot.id,
            branch=self.branches.current_branch,
        )

        return snapshot

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore world state from a snapshot."""
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            logger.warning("snapshot_not_found", snapshot_id=snapshot_id)
            return False

        # Clone the snapshot state to avoid mutation
        restored = snapshot.world_state.clone()
        self.world_engine.set_state(restored)

        # Restore RNG state if available
        if restored.rng_state is not None:
            self.world_engine.restore_rng_state(restored.rng_state)

        # Switch branch if snapshot is on a different branch
        if snapshot.branch_name and snapshot.branch_name != self.branches.current_branch:
            try:
                self.branches.switch(snapshot.branch_name)
            except ValueError:
                pass

        logger.info(
            "snapshot_restored",
            tick=snapshot.tick,
            snapshot_id=snapshot_id,
        )
        return True

    def get_snapshot(self, snapshot_id: str) -> Optional[TimelineSnapshot]:
        return self.snapshots.get(snapshot_id)

    def get_snapshots_in_range(
        self,
        start_tick: int,
        end_tick: int,
    ) -> List[TimelineSnapshot]:
        return self.snapshots.get_range(start_tick, end_tick)

    # -- Branching --

    def create_branch(
        self,
        branch_name: str,
        from_snapshot: Optional[str] = None,
    ) -> bool:
        """Create a new timeline branch."""
        try:
            # Restore snapshot if specified
            if from_snapshot:
                if not self.restore_snapshot(from_snapshot):
                    return False

            # Create branch snapshot
            branch_snapshot = self.create_snapshot(
                description=f"Branch point for {branch_name}",
            )

            # Create the branch
            self.branches.create(
                branch_name,
                from_snapshot_id=branch_snapshot.id,
            )
            self.branches.switch(branch_name)

            logger.info("branch_created", name=branch_name)
            return True

        except ValueError as exc:
            logger.warning("branch_creation_failed", error=str(exc))
            return False

    def switch_branch(self, branch_name: str) -> bool:
        """Switch to a branch, restoring its latest snapshot."""
        try:
            branch = self.branches.switch(branch_name)
        except ValueError:
            logger.warning("branch_not_found", name=branch_name)
            return False

        latest = self.branches.latest_snapshot(branch_name)
        if latest:
            self.restore_snapshot(latest)

        return True

    def list_branches(self) -> List[str]:
        return self.branches.list_branches()

    def get_branch_history(
        self,
        branch_name: Optional[str] = None,
    ) -> List[TimelineSnapshot]:
        """Get all snapshots on a branch."""
        ids = self.branches.get_snapshots(branch_name)
        return [
            self.snapshots.get(sid)
            for sid in ids
            if self.snapshots.get(sid) is not None
        ]

    # -- Time Travel --

    def rewind(self, ticks: int) -> bool:
        """Rewind simulation by specified ticks."""
        target_tick = max(0, self.world_engine.state.tick - ticks)

        snapshot = self.snapshots.get_nearest(target_tick, direction="before")
        if snapshot is None:
            logger.warning("no_snapshot_for_rewind", target_tick=target_tick)
            return False

        return self.restore_snapshot(snapshot.id)

    def fast_forward(self, ticks: int) -> bool:
        """Fast-forward to a future snapshot (if exists)."""
        target_tick = self.world_engine.state.tick + ticks

        snapshot = self.snapshots.get_nearest(target_tick, direction="after")
        if snapshot is None:
            logger.warning("no_snapshot_for_fast_forward", target_tick=target_tick)
            return False

        return self.restore_snapshot(snapshot.id)

    def goto_tick(self, tick: int) -> bool:
        """Jump to the nearest snapshot at or before a specific tick."""
        snapshot = self.snapshots.get_nearest(tick, direction="before")
        if snapshot is None:
            return False
        return self.restore_snapshot(snapshot.id)

    # -- Comparison --

    def compare_snapshots(
        self,
        snapshot_id_1: str,
        snapshot_id_2: str,
    ) -> Dict[str, Any]:
        """Compare two snapshots."""
        s1 = self.snapshots.get(snapshot_id_1)
        s2 = self.snapshots.get(snapshot_id_2)

        if not s1 or not s2:
            return {"error": "Snapshot not found"}

        w1 = s1.world_state
        w2 = s2.world_state

        entities_1 = set(w1.entities.keys())
        entities_2 = set(w2.entities.keys())

        # Find modified entities (by fingerprint)
        common = entities_1 & entities_2
        modified = [
            eid for eid in common
            if w1.entities[eid].fingerprint() != w2.entities[eid].fingerprint()
        ]

        return {
            "tick_diff": s2.tick - s1.tick,
            "time_diff": s2.simulation_time - s1.simulation_time,
            "entity_count_diff": len(entities_2) - len(entities_1),
            "entities_added": list(entities_2 - entities_1),
            "entities_removed": list(entities_1 - entities_2),
            "entities_modified": modified,
            "metrics_diff": {
                k: w2.metrics.get(k, 0) - w1.metrics.get(k, 0)
                for k in set(w1.metrics) | set(w2.metrics)
            },
            "fingerprint_1": s1.state_fingerprint or w1.fingerprint(),
            "fingerprint_2": s2.state_fingerprint or w2.fingerprint(),
        }

    def compare_branches(
        self,
        branch_1: str,
        branch_2: str,
    ) -> Dict[str, Any]:
        """Compare the latest states of two branches."""
        snap1 = self.branches.latest_snapshot(branch_1)
        snap2 = self.branches.latest_snapshot(branch_2)

        if not snap1 or not snap2:
            return {"error": "Branch not found or empty"}

        return self.compare_snapshots(snap1, snap2)

    # -- Recording --

    def start_recording(self) -> None:
        self.replay.start_recording()

    def stop_recording(self) -> Any:
        return self.replay.stop_recording()

    async def replay_events(
        self,
        events: List[SimulationEvent],
        from_snapshot: Optional[str] = None,
    ) -> None:
        """Replay events, optionally from a snapshot."""
        if from_snapshot:
            self.restore_snapshot(from_snapshot)
        await self.replay.replay_events(events)

    # -- Stats --

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "snapshot_count": self.snapshots.count,
            "branch_count": self.branches.branch_count,
            "current_branch": self.branches.current_branch,
            "tick_range": self.snapshots.tick_range,
            "replay_sessions": len(self.replay.list_sessions()),
        }
