"""
AION Consensus State Machine

Applies committed Raft log entries to the application state.
Implements a deterministic state machine that processes commands
from the replicated log, maintaining a key-value state that
can be snapshot and restored for log compaction.

Supported command types:
- set_state / delete_state: Key-value state manipulation
- add_node / remove_node: Cluster membership changes
- update_config: Configuration updates
- task_assign / task_complete: Distributed task lifecycle
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from aion.distributed.types import RaftLogEntry, SnapshotMetadata

logger = structlog.get_logger(__name__)


class ConsensusStateMachine:
    """
    Deterministic state machine for the Raft consensus protocol.

    Applies committed log entries in order, producing deterministic
    state transitions. All nodes applying the same log sequence
    arrive at the same state.

    Features:
    - Command handler dispatch by command type
    - Pluggable custom command handlers
    - Snapshot/restore for log compaction
    - State integrity verification via checksums
    - Applied index tracking to prevent duplicate application
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="consensus_state_machine")

        # Core key-value state
        self._state: Dict[str, Any] = {}

        # Cluster membership state
        self._cluster_nodes: Dict[str, Dict[str, Any]] = {}

        # Configuration state
        self._config: Dict[str, Any] = {}

        # Task tracking state
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._completed_tasks: Dict[str, Dict[str, Any]] = {}

        # Applied index tracking to prevent re-application
        self._last_applied_index: int = -1
        self._last_applied_term: int = 0

        # Metrics
        self._commands_applied: int = 0
        self._snapshots_taken: int = 0
        self._snapshots_restored: int = 0

        # Custom command handlers
        self._custom_handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

        # Built-in command dispatch table
        self._dispatch: Dict[str, Callable[[RaftLogEntry], Any]] = {
            "set_state": self._handle_set_state,
            "delete_state": self._handle_delete_state,
            "add_node": self._handle_add_node,
            "remove_node": self._handle_remove_node,
            "update_config": self._handle_update_config,
            "task_assign": self._handle_task_assign,
            "task_complete": self._handle_task_complete,
            "noop": self._handle_noop,
        }

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def last_applied_index(self) -> int:
        """Index of the last applied log entry."""
        return self._last_applied_index

    @property
    def last_applied_term(self) -> int:
        """Term of the last applied log entry."""
        return self._last_applied_term

    @property
    def state(self) -> Dict[str, Any]:
        """Read-only view of current key-value state."""
        return dict(self._state)

    @property
    def cluster_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Read-only view of cluster membership state."""
        return dict(self._cluster_nodes)

    @property
    def config(self) -> Dict[str, Any]:
        """Read-only view of configuration state."""
        return dict(self._config)

    @property
    def commands_applied(self) -> int:
        """Total number of commands applied."""
        return self._commands_applied

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def apply(self, entry: RaftLogEntry) -> Any:
        """
        Apply a committed log entry to the state machine.

        Entries must be applied in strict index order. Duplicate
        applications (entries at or before last_applied_index) are
        silently ignored to support idempotent replay.

        Args:
            entry: The committed Raft log entry to apply.

        Returns:
            The result of applying the command, or None for no-ops.
        """
        # Guard against out-of-order or duplicate application
        if entry.index <= self._last_applied_index:
            self._log.debug(
                "skipping_already_applied_entry",
                entry_index=entry.index,
                last_applied=self._last_applied_index,
            )
            return None

        # Handle noop entries (leader establishment)
        if entry.is_noop:
            self._last_applied_index = entry.index
            self._last_applied_term = entry.term
            self._commands_applied += 1
            self._log.debug("applied_noop", index=entry.index, term=entry.term)
            return None

        # Dispatch to command handler
        handler = self._dispatch.get(entry.command)
        if handler is None:
            # Check custom handlers
            custom = self._custom_handlers.get(entry.command)
            if custom is not None:
                result = custom(entry.data)
            else:
                self._log.warning(
                    "unknown_command",
                    command=entry.command,
                    index=entry.index,
                )
                result = None
        else:
            result = handler(entry)

        # Update applied tracking
        self._last_applied_index = entry.index
        self._last_applied_term = entry.term
        self._commands_applied += 1

        self._log.debug(
            "entry_applied",
            command=entry.command,
            index=entry.index,
            term=entry.term,
        )
        return result

    async def apply_batch(self, entries: List[RaftLogEntry]) -> List[Any]:
        """
        Apply a batch of committed entries in order.

        Args:
            entries: Ordered list of entries to apply.

        Returns:
            List of results from each applied entry.
        """
        results: List[Any] = []
        for entry in entries:
            result = await self.apply(entry)
            results.append(result)
        return results

    def register_handler(
        self, command: str, handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """
        Register a custom command handler.

        Custom handlers receive the entry's data dict and return
        an arbitrary result.

        Args:
            command: The command name to handle.
            handler: Callable that processes the command data.
        """
        self._custom_handlers[command] = handler
        self._log.info("custom_handler_registered", command=command)

    def unregister_handler(self, command: str) -> None:
        """Remove a custom command handler."""
        self._custom_handlers.pop(command, None)

    def get_value(self, key: str) -> Optional[Any]:
        """Get a value from the state machine's key-value store."""
        return self._state.get(key)

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the state machine."""
        return key in self._state

    async def take_snapshot(self) -> bytes:
        """
        Serialize the full state machine state for snapshotting.

        Returns:
            Serialized state as bytes.
        """
        snapshot_data = {
            "version": 1,
            "state": self._state,
            "cluster_nodes": self._cluster_nodes,
            "config": self._config,
            "tasks": self._tasks,
            "completed_tasks": self._completed_tasks,
            "last_applied_index": self._last_applied_index,
            "last_applied_term": self._last_applied_term,
            "commands_applied": self._commands_applied,
            "timestamp": datetime.now().isoformat(),
        }

        data = json.dumps(snapshot_data, sort_keys=True, default=str).encode("utf-8")
        self._snapshots_taken += 1

        checksum = hashlib.sha256(data).hexdigest()
        self._log.info(
            "snapshot_taken",
            size_bytes=len(data),
            last_applied_index=self._last_applied_index,
            checksum=checksum,
        )
        return data

    async def restore_snapshot(self, data: bytes) -> None:
        """
        Restore state machine from a snapshot.

        Completely replaces all current state with the snapshot
        contents. Used during log compaction and when a follower
        falls too far behind.

        Args:
            data: Serialized snapshot bytes.
        """
        snapshot_data = json.loads(data.decode("utf-8"))
        version = snapshot_data.get("version", 1)

        if version != 1:
            self._log.error("unsupported_snapshot_version", version=version)
            raise ValueError(f"Unsupported snapshot version: {version}")

        self._state = snapshot_data.get("state", {})
        self._cluster_nodes = snapshot_data.get("cluster_nodes", {})
        self._config = snapshot_data.get("config", {})
        self._tasks = snapshot_data.get("tasks", {})
        self._completed_tasks = snapshot_data.get("completed_tasks", {})
        self._last_applied_index = snapshot_data.get("last_applied_index", -1)
        self._last_applied_term = snapshot_data.get("last_applied_term", 0)
        self._commands_applied = snapshot_data.get("commands_applied", 0)

        self._snapshots_restored += 1

        self._log.info(
            "snapshot_restored",
            last_applied_index=self._last_applied_index,
            state_keys=len(self._state),
            cluster_nodes=len(self._cluster_nodes),
        )

    def get_snapshot_metadata(self) -> SnapshotMetadata:
        """Build snapshot metadata for the current state."""
        return SnapshotMetadata(
            last_included_index=self._last_applied_index,
            last_included_term=self._last_applied_term,
            node_count=len(self._cluster_nodes),
        )

    def compute_state_checksum(self) -> str:
        """
        Compute a checksum of the current state for integrity verification.

        All nodes with the same applied log should produce the same checksum.
        """
        state_repr = json.dumps(
            {
                "state": self._state,
                "cluster_nodes": self._cluster_nodes,
                "config": self._config,
                "tasks": self._tasks,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(state_repr.encode("utf-8")).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get state machine statistics."""
        return {
            "last_applied_index": self._last_applied_index,
            "last_applied_term": self._last_applied_term,
            "commands_applied": self._commands_applied,
            "state_keys": len(self._state),
            "cluster_nodes": len(self._cluster_nodes),
            "config_keys": len(self._config),
            "active_tasks": len(self._tasks),
            "completed_tasks": len(self._completed_tasks),
            "snapshots_taken": self._snapshots_taken,
            "snapshots_restored": self._snapshots_restored,
            "custom_handlers": list(self._custom_handlers.keys()),
        }

    # -------------------------------------------------------------------------
    # Built-in Command Handlers
    # -------------------------------------------------------------------------

    def _handle_noop(self, entry: RaftLogEntry) -> None:
        """Handle no-op entries used for leader establishment."""
        return None

    def _handle_set_state(self, entry: RaftLogEntry) -> Any:
        """
        Set a key-value pair in state.

        Expected data: {"key": str, "value": Any}
        Optional: {"ttl": int} for time-to-live (stored but not enforced here)
        """
        data = entry.data
        key = data.get("key", "")
        value = data.get("value")

        if not key:
            self._log.warning("set_state_missing_key", index=entry.index)
            return None

        old_value = self._state.get(key)
        self._state[key] = value

        self._log.debug(
            "state_set",
            key=key,
            had_previous=old_value is not None,
            index=entry.index,
        )
        return value

    def _handle_delete_state(self, entry: RaftLogEntry) -> Any:
        """
        Delete a key from state.

        Expected data: {"key": str}
        """
        data = entry.data
        key = data.get("key", "")

        if not key:
            self._log.warning("delete_state_missing_key", index=entry.index)
            return None

        old_value = self._state.pop(key, None)

        self._log.debug(
            "state_deleted",
            key=key,
            existed=old_value is not None,
            index=entry.index,
        )
        return old_value

    def _handle_add_node(self, entry: RaftLogEntry) -> Dict[str, Any]:
        """
        Add a node to the cluster membership.

        Expected data: {"node_id": str, "host": str, "port": int, ...}
        """
        data = entry.data
        node_id = data.get("node_id", "")

        if not node_id:
            self._log.warning("add_node_missing_id", index=entry.index)
            return {}

        node_record = {
            "node_id": node_id,
            "host": data.get("host", ""),
            "port": data.get("port", 0),
            "role": data.get("role", "follower"),
            "capabilities": data.get("capabilities", []),
            "added_at": datetime.now().isoformat(),
            "added_at_index": entry.index,
        }

        self._cluster_nodes[node_id] = node_record

        self._log.info(
            "node_added",
            node_id=node_id,
            host=node_record["host"],
            index=entry.index,
        )
        return node_record

    def _handle_remove_node(self, entry: RaftLogEntry) -> Optional[Dict[str, Any]]:
        """
        Remove a node from the cluster membership.

        Expected data: {"node_id": str}
        """
        data = entry.data
        node_id = data.get("node_id", "")

        if not node_id:
            self._log.warning("remove_node_missing_id", index=entry.index)
            return None

        removed = self._cluster_nodes.pop(node_id, None)

        if removed:
            self._log.info("node_removed", node_id=node_id, index=entry.index)
        else:
            self._log.warning(
                "remove_node_not_found", node_id=node_id, index=entry.index
            )
        return removed

    def _handle_update_config(self, entry: RaftLogEntry) -> Dict[str, Any]:
        """
        Update cluster configuration.

        Expected data: {"key": str, "value": Any}
        Supports nested keys with dot notation: "raft.election_timeout_min_ms"
        """
        data = entry.data
        key = data.get("key", "")
        value = data.get("value")

        if not key:
            self._log.warning("update_config_missing_key", index=entry.index)
            return {}

        # Support nested key with dot notation
        parts = key.split(".")
        target = self._config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        old_value = target.get(parts[-1])
        target[parts[-1]] = value

        self._log.info(
            "config_updated",
            key=key,
            had_previous=old_value is not None,
            index=entry.index,
        )
        return {"key": key, "old_value": old_value, "new_value": value}

    def _handle_task_assign(self, entry: RaftLogEntry) -> Dict[str, Any]:
        """
        Record a task assignment.

        Expected data: {"task_id": str, "node_id": str, "task_type": str, ...}
        """
        data = entry.data
        task_id = data.get("task_id", "")

        if not task_id:
            self._log.warning("task_assign_missing_id", index=entry.index)
            return {}

        task_record = {
            "task_id": task_id,
            "node_id": data.get("node_id", ""),
            "task_type": data.get("task_type", ""),
            "payload": data.get("payload", {}),
            "priority": data.get("priority", 2),
            "status": "assigned",
            "assigned_at": datetime.now().isoformat(),
            "assigned_at_index": entry.index,
        }

        self._tasks[task_id] = task_record

        self._log.debug(
            "task_assigned",
            task_id=task_id,
            node_id=task_record["node_id"],
            index=entry.index,
        )
        return task_record

    def _handle_task_complete(self, entry: RaftLogEntry) -> Optional[Dict[str, Any]]:
        """
        Record a task completion.

        Expected data: {"task_id": str, "result": Any, "status": str}
        """
        data = entry.data
        task_id = data.get("task_id", "")

        if not task_id:
            self._log.warning("task_complete_missing_id", index=entry.index)
            return None

        task_record = self._tasks.pop(task_id, None)
        if task_record is None:
            self._log.warning(
                "task_complete_not_found", task_id=task_id, index=entry.index
            )
            return None

        task_record["status"] = data.get("status", "completed")
        task_record["result"] = data.get("result")
        task_record["error"] = data.get("error")
        task_record["completed_at"] = datetime.now().isoformat()
        task_record["completed_at_index"] = entry.index

        self._completed_tasks[task_id] = task_record

        self._log.debug(
            "task_completed",
            task_id=task_id,
            status=task_record["status"],
            index=entry.index,
        )
        return task_record
