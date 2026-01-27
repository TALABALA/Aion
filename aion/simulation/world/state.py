"""AION World State Manager - Efficient state management with COW snapshots.

Provides:
- WorldStateManager: Manages world state with copy-on-write semantics.
- Incremental state diffs for efficient storage.
- State validation and integrity checks.
- Undo/redo via state stack.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import structlog

from aion.simulation.types import Entity, WorldState

logger = structlog.get_logger(__name__)


@dataclass
class StateDiff:
    """Incremental diff between two world states.

    Captures only what changed for efficient storage and transmission.
    """

    from_tick: int = 0
    to_tick: int = 0

    # Entity changes
    entities_added: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    entities_removed: Set[str] = field(default_factory=set)
    entities_modified: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Global state changes
    global_state_changes: Dict[str, Any] = field(default_factory=dict)

    # Metric changes
    metric_changes: Dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return (
            not self.entities_added
            and not self.entities_removed
            and not self.entities_modified
            and not self.global_state_changes
            and not self.metric_changes
        )


class WorldStateManager:
    """Manages world state with COW snapshots and incremental diffs.

    Features:
    - Copy-on-write entity mutation tracking.
    - State diff computation for efficient storage.
    - Undo/redo stack.
    - State integrity validation.
    - Metrics aggregation.
    """

    def __init__(self, max_undo_depth: int = 100) -> None:
        self._state = WorldState()
        self._max_undo_depth = max_undo_depth

        # Undo/redo
        self._undo_stack: Deque[WorldState] = deque(maxlen=max_undo_depth)
        self._redo_stack: Deque[WorldState] = deque(maxlen=max_undo_depth)

        # Diff tracking
        self._previous_fingerprints: Dict[str, str] = {}

        # Validators
        self._validators: List[Any] = []

    @property
    def state(self) -> WorldState:
        return self._state

    @state.setter
    def state(self, new_state: WorldState) -> None:
        self._state = new_state

    # -- COW Mutation --

    def begin_transaction(self) -> None:
        """Save current state for potential undo."""
        self._undo_stack.append(self._state.clone())
        self._redo_stack.clear()

    def commit(self) -> None:
        """Commit the current state (clear dirty tracking)."""
        self._state._dirty_entities.clear()
        self._update_fingerprints()

    def rollback(self) -> bool:
        """Rollback to previous state."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._state.clone())
        self._state = self._undo_stack.pop()
        return True

    def undo(self) -> bool:
        """Undo last state change."""
        return self.rollback()

    def redo(self) -> bool:
        """Redo last undone change."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._state.clone())
        self._state = self._redo_stack.pop()
        return True

    # -- Diff Computation --

    def compute_diff(self, old_state: WorldState, new_state: WorldState) -> StateDiff:
        """Compute the diff between two world states."""
        diff = StateDiff(from_tick=old_state.tick, to_tick=new_state.tick)

        old_ids = set(old_state.entities.keys())
        new_ids = set(new_state.entities.keys())

        # Added entities
        for eid in new_ids - old_ids:
            diff.entities_added[eid] = new_state.entities[eid].to_dict()

        # Removed entities
        diff.entities_removed = old_ids - new_ids

        # Modified entities (use fingerprints for efficiency)
        for eid in old_ids & new_ids:
            old_entity = old_state.entities[eid]
            new_entity = new_state.entities[eid]
            if old_entity.fingerprint() != new_entity.fingerprint():
                diff.entities_modified[eid] = {
                    "properties": new_entity.properties,
                    "state": new_entity.state,
                    "components": new_entity.components,
                    "version": new_entity.version,
                }

        # Global state changes
        for key in set(old_state.global_state) | set(new_state.global_state):
            old_val = old_state.global_state.get(key)
            new_val = new_state.global_state.get(key)
            if old_val != new_val:
                diff.global_state_changes[key] = new_val

        # Metric changes
        for key in set(old_state.metrics) | set(new_state.metrics):
            old_val = old_state.metrics.get(key, 0.0)
            new_val = new_state.metrics.get(key, 0.0)
            if old_val != new_val:
                diff.metric_changes[key] = new_val

        return diff

    def apply_diff(self, state: WorldState, diff: StateDiff) -> WorldState:
        """Apply a diff to a world state, returning a new state."""
        new_state = state.clone()

        # Remove entities
        for eid in diff.entities_removed:
            new_state.entities.pop(eid, None)

        # Add entities
        for eid, edata in diff.entities_added.items():
            entity = Entity(
                id=eid,
                name=edata.get("name", ""),
                properties=edata.get("properties", {}),
                state=edata.get("state", {}),
                components=edata.get("components", {}),
            )
            new_state.entities[eid] = entity

        # Modify entities
        for eid, changes in diff.entities_modified.items():
            entity = new_state.mutate_entity(eid)
            if entity:
                if "properties" in changes:
                    entity.properties = changes["properties"]
                if "state" in changes:
                    entity.state = changes["state"]
                if "components" in changes:
                    entity.components = changes["components"]

        # Global state
        for key, value in diff.global_state_changes.items():
            if value is None:
                new_state.global_state.pop(key, None)
            else:
                new_state.global_state[key] = value

        # Metrics
        for key, value in diff.metric_changes.items():
            new_state.metrics[key] = value

        new_state.tick = diff.to_tick
        return new_state

    # -- Validation --

    def validate(self) -> List[str]:
        """Validate current state integrity.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[str] = []

        # Check entity relationships point to existing entities
        for eid, entity in self._state.entities.items():
            if entity.parent_id and entity.parent_id not in self._state.entities:
                errors.append(
                    f"Entity {eid} has dangling parent_id: {entity.parent_id}"
                )
            for child_id in entity.children_ids:
                if child_id not in self._state.entities:
                    errors.append(
                        f"Entity {eid} has dangling child_id: {child_id}"
                    )
            for rel_name, rel_ids in entity.relationships.items():
                for rid in rel_ids:
                    if rid not in self._state.entities:
                        errors.append(
                            f"Entity {eid} has dangling relationship "
                            f"{rel_name} -> {rid}"
                        )

        # Run custom validators
        for validator in self._validators:
            try:
                validator_errors = validator(self._state)
                if validator_errors:
                    errors.extend(validator_errors)
            except Exception as exc:
                errors.append(f"Validator error: {exc}")

        return errors

    def add_validator(self, validator: Any) -> None:
        """Add a state validator function.

        Validator signature: (WorldState) -> List[str]
        """
        self._validators.append(validator)

    # -- Metrics --

    def update_metric(self, name: str, value: float) -> None:
        """Update a metric value."""
        self._state.metrics[name] = value

    def increment_metric(self, name: str, delta: float = 1.0) -> float:
        """Increment a metric, returning new value."""
        current = self._state.metrics.get(name, 0.0)
        new_val = current + delta
        self._state.metrics[name] = new_val
        return new_val

    def get_metric(self, name: str, default: float = 0.0) -> float:
        return self._state.metrics.get(name, default)

    # -- Snapshot --

    def snapshot(self) -> WorldState:
        """Create a COW snapshot of current state."""
        return self._state.clone()

    def restore(self, state: WorldState) -> None:
        """Restore from a snapshot."""
        self._undo_stack.append(self._state.clone())
        self._state = state.clone()

    # -- Internal --

    def _update_fingerprints(self) -> None:
        """Update entity fingerprints for change detection."""
        self._previous_fingerprints = {
            eid: entity.fingerprint()
            for eid, entity in self._state.entities.items()
        }

    def get_changed_entities(self) -> Set[str]:
        """Get IDs of entities changed since last commit."""
        changed = set()
        for eid, entity in self._state.entities.items():
            prev_fp = self._previous_fingerprints.get(eid)
            if prev_fp is None or entity.fingerprint() != prev_fp:
                changed.add(eid)
        # Also include removed entities
        for eid in self._previous_fingerprints:
            if eid not in self._state.entities:
                changed.add(eid)
        return changed
