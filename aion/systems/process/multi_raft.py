"""
AION Multi-Raft Implementation

Multi-Raft for scalable distributed systems:
- Multiple Raft groups (one per partition/shard)
- Consistent hashing for partition assignment
- Cross-group coordination
- Load balancing across groups
- Similar to TiKV/CockroachDB architecture
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog

from aion.systems.process.consensus import (
    RaftNode,
    RaftState,
    RaftLog,
    LogEntry,
    NotLeaderError,
)

logger = structlog.get_logger(__name__)


class PartitionState(Enum):
    """State of a partition."""
    ACTIVE = auto()
    SPLITTING = auto()
    MERGING = auto()
    MIGRATING = auto()
    INACTIVE = auto()


@dataclass
class PartitionInfo:
    """Information about a partition."""
    partition_id: str
    group_id: str
    start_key: bytes
    end_key: bytes
    state: PartitionState = PartitionState.ACTIVE
    replicas: Set[str] = field(default_factory=set)
    leader: Optional[str] = None
    epoch: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def contains_key(self, key: bytes) -> bool:
        """Check if this partition contains the given key."""
        return self.start_key <= key < self.end_key


@dataclass
class RaftGroupConfig:
    """Configuration for a Raft group."""
    group_id: str
    partition_id: str
    members: Set[str]
    election_timeout_range: Tuple[float, float] = (0.15, 0.3)
    heartbeat_interval: float = 0.05


class RaftGroup:
    """
    A single Raft group managing one partition.

    Each group is an independent Raft cluster that handles
    a subset of the keyspace.
    """

    def __init__(
        self,
        group_id: str,
        node_id: str,
        config: RaftGroupConfig,
        apply_callback: Callable[[str, Dict[str, Any]], Any],
        storage_path: Optional[str] = None,
    ):
        self.group_id = group_id
        self.node_id = node_id
        self.config = config
        self.apply_callback = apply_callback

        # Create the underlying Raft node
        peers = config.members - {node_id}
        self.raft_node = RaftNode(
            node_id=node_id,
            peers=list(peers),
            apply_callback=lambda cmd: apply_callback(group_id, cmd),
            storage_path=storage_path,
            election_timeout_range=config.election_timeout_range,
            heartbeat_interval=config.heartbeat_interval,
        )

        # Group metadata
        self.partition_id = config.partition_id
        self._active = False

        # Stats
        self._stats = {
            "proposals_received": 0,
            "proposals_committed": 0,
            "reads_served": 0,
        }

    async def start(self) -> None:
        """Start the Raft group."""
        await self.raft_node.start()
        self._active = True
        logger.info(f"Raft group {self.group_id} started")

    async def stop(self) -> None:
        """Stop the Raft group."""
        self._active = False
        await self.raft_node.stop()
        logger.info(f"Raft group {self.group_id} stopped")

    async def propose(self, command: Dict[str, Any], timeout: float = 5.0) -> Any:
        """Propose a command to the group."""
        self._stats["proposals_received"] += 1
        result = await self.raft_node.propose(command, timeout)
        self._stats["proposals_committed"] += 1
        return result

    async def read(self, key: str) -> Any:
        """Read a value from the group's state."""
        self._stats["reads_served"] += 1
        return await self.raft_node.read(key)

    def is_leader(self) -> bool:
        """Check if this node is the leader of this group."""
        return self.raft_node.state == RaftState.LEADER

    def get_leader(self) -> Optional[str]:
        """Get the current leader of this group."""
        return self.raft_node.leader_id

    def get_stats(self) -> Dict[str, Any]:
        """Get group statistics."""
        return {
            **self._stats,
            **self.raft_node.get_stats(),
            "group_id": self.group_id,
            "partition_id": self.partition_id,
        }


class MultiRaftRouter:
    """
    Routes requests to the appropriate Raft group.

    Uses consistent hashing to determine which partition
    handles a given key.
    """

    def __init__(self, hash_slots: int = 16384):
        self.hash_slots = hash_slots
        self._partitions: Dict[str, PartitionInfo] = {}
        self._slot_to_partition: Dict[int, str] = {}

    def _hash_key(self, key: bytes) -> int:
        """Hash a key to a slot number."""
        h = hashlib.sha256(key).digest()
        return int.from_bytes(h[:4], 'big') % self.hash_slots

    def add_partition(self, partition: PartitionInfo) -> None:
        """Add a partition to the router."""
        self._partitions[partition.partition_id] = partition

        # Assign slots to this partition
        start_slot = self._hash_key(partition.start_key)
        end_slot = self._hash_key(partition.end_key)

        if start_slot <= end_slot:
            for slot in range(start_slot, end_slot):
                self._slot_to_partition[slot] = partition.partition_id
        else:
            # Wraps around
            for slot in range(start_slot, self.hash_slots):
                self._slot_to_partition[slot] = partition.partition_id
            for slot in range(0, end_slot):
                self._slot_to_partition[slot] = partition.partition_id

    def remove_partition(self, partition_id: str) -> None:
        """Remove a partition from the router."""
        if partition_id in self._partitions:
            del self._partitions[partition_id]
            self._slot_to_partition = {
                slot: pid for slot, pid in self._slot_to_partition.items()
                if pid != partition_id
            }

    def get_partition_for_key(self, key: bytes) -> Optional[str]:
        """Get the partition ID that handles a given key."""
        slot = self._hash_key(key)
        return self._slot_to_partition.get(slot)

    def get_partition_info(self, partition_id: str) -> Optional[PartitionInfo]:
        """Get information about a partition."""
        return self._partitions.get(partition_id)

    def get_all_partitions(self) -> List[PartitionInfo]:
        """Get all partitions."""
        return list(self._partitions.values())


class MultiRaftNode:
    """
    A node in a Multi-Raft cluster.

    Manages multiple Raft groups, one per partition.
    Similar to a TiKV store or CockroachDB node.
    """

    def __init__(
        self,
        node_id: str,
        storage_path: Optional[str] = None,
        max_groups: int = 100,
    ):
        self.node_id = node_id
        self.storage_path = storage_path
        self.max_groups = max_groups

        # Raft groups
        self._groups: Dict[str, RaftGroup] = {}
        self._router = MultiRaftRouter()

        # State machine (in-memory for simplicity)
        self._state: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Cross-group transaction support
        self._pending_txns: Dict[str, "DistributedTransaction"] = {}

        # Background tasks
        self._shutdown = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Stats
        self._stats = {
            "groups_created": 0,
            "groups_removed": 0,
            "requests_routed": 0,
            "cross_group_txns": 0,
        }

    async def start(self) -> None:
        """Start the Multi-Raft node."""
        self._heartbeat_task = asyncio.create_task(self._group_heartbeat_loop())
        logger.info(f"Multi-Raft node {self.node_id} started")

    async def stop(self) -> None:
        """Stop the Multi-Raft node."""
        self._shutdown = True

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Stop all groups
        for group in self._groups.values():
            await group.stop()

        logger.info(f"Multi-Raft node {self.node_id} stopped")

    async def create_group(self, config: RaftGroupConfig) -> RaftGroup:
        """Create a new Raft group."""
        if len(self._groups) >= self.max_groups:
            raise RuntimeError(f"Maximum groups ({self.max_groups}) reached")

        if config.group_id in self._groups:
            raise ValueError(f"Group {config.group_id} already exists")

        group = RaftGroup(
            group_id=config.group_id,
            node_id=self.node_id,
            config=config,
            apply_callback=self._apply_to_state,
            storage_path=f"{self.storage_path}/{config.group_id}" if self.storage_path else None,
        )

        await group.start()
        self._groups[config.group_id] = group
        self._stats["groups_created"] += 1

        return group

    async def remove_group(self, group_id: str) -> None:
        """Remove a Raft group."""
        if group_id in self._groups:
            await self._groups[group_id].stop()
            del self._groups[group_id]
            self._stats["groups_removed"] += 1

    def get_group(self, group_id: str) -> Optional[RaftGroup]:
        """Get a Raft group by ID."""
        return self._groups.get(group_id)

    async def propose(
        self,
        key: bytes,
        command: Dict[str, Any],
        timeout: float = 5.0,
    ) -> Any:
        """Propose a command to the appropriate group."""
        partition_id = self._router.get_partition_for_key(key)
        if not partition_id:
            raise ValueError(f"No partition found for key")

        partition = self._router.get_partition_info(partition_id)
        if not partition:
            raise ValueError(f"Partition {partition_id} not found")

        group = self._groups.get(partition.group_id)
        if not group:
            raise ValueError(f"Group {partition.group_id} not found on this node")

        if not group.is_leader():
            raise NotLeaderError(group.get_leader())

        self._stats["requests_routed"] += 1
        return await group.propose(command, timeout)

    async def read(self, key: bytes) -> Any:
        """Read a value from the appropriate group."""
        partition_id = self._router.get_partition_for_key(key)
        if not partition_id:
            return None

        partition = self._router.get_partition_info(partition_id)
        if not partition:
            return None

        group = self._groups.get(partition.group_id)
        if not group:
            return None

        self._stats["requests_routed"] += 1
        return await group.read(key.decode())

    def _apply_to_state(self, group_id: str, command: Dict[str, Any]) -> Any:
        """Apply a command to the state machine."""
        cmd_type = command.get("type", "")

        if cmd_type == "set":
            key = command.get("key")
            value = command.get("value")
            self._state[group_id][key] = value
            return value

        elif cmd_type == "delete":
            key = command.get("key")
            return self._state[group_id].pop(key, None)

        elif cmd_type == "prepare":
            # Two-phase commit prepare
            return self._handle_prepare(group_id, command)

        elif cmd_type == "commit":
            # Two-phase commit commit
            return self._handle_commit(group_id, command)

        elif cmd_type == "abort":
            # Two-phase commit abort
            return self._handle_abort(group_id, command)

        return None

    async def _group_heartbeat_loop(self) -> None:
        """Monitor group health."""
        while not self._shutdown:
            try:
                await asyncio.sleep(1.0)

                for group_id, group in list(self._groups.items()):
                    if not group._active:
                        continue

                    # Check group health
                    stats = group.get_stats()

                    # Log any issues
                    if stats.get("state") == "LEADER" and stats.get("commit_index", 0) > stats.get("last_applied", 0) + 1000:
                        logger.warning(f"Group {group_id} has large apply backlog")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    # === Two-Phase Commit for Cross-Group Transactions ===

    def _handle_prepare(self, group_id: str, command: Dict[str, Any]) -> bool:
        """Handle prepare phase of 2PC."""
        txn_id = command.get("txn_id")
        operations = command.get("operations", [])

        # Check if we can execute all operations
        for op in operations:
            if op.get("group_id") == group_id:
                # Validate operation
                key = op.get("key")
                if op.get("type") == "delete" and key not in self._state[group_id]:
                    return False  # Cannot delete non-existent key

        # Lock resources (simplified - in production use proper locking)
        return True

    def _handle_commit(self, group_id: str, command: Dict[str, Any]) -> Any:
        """Handle commit phase of 2PC."""
        txn_id = command.get("txn_id")
        operations = command.get("operations", [])

        results = []
        for op in operations:
            if op.get("group_id") == group_id:
                if op.get("type") == "set":
                    self._state[group_id][op["key"]] = op["value"]
                    results.append(op["value"])
                elif op.get("type") == "delete":
                    results.append(self._state[group_id].pop(op["key"], None))

        return results

    def _handle_abort(self, group_id: str, command: Dict[str, Any]) -> None:
        """Handle abort of 2PC."""
        txn_id = command.get("txn_id")
        # Release any locks (simplified)
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        group_stats = {
            gid: group.get_stats()
            for gid, group in self._groups.items()
        }

        return {
            **self._stats,
            "node_id": self.node_id,
            "active_groups": len(self._groups),
            "leader_groups": sum(1 for g in self._groups.values() if g.is_leader()),
            "groups": group_stats,
        }


# === Distributed Transactions ===

class TransactionState(Enum):
    """State of a distributed transaction."""
    PENDING = auto()
    PREPARING = auto()
    PREPARED = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ABORTING = auto()
    ABORTED = auto()


@dataclass
class TransactionOperation:
    """A single operation in a transaction."""
    op_type: str  # set, delete, read
    key: bytes
    value: Any = None
    group_id: Optional[str] = None


@dataclass
class DistributedTransaction:
    """A distributed transaction spanning multiple Raft groups."""
    txn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operations: List[TransactionOperation] = field(default_factory=list)
    state: TransactionState = TransactionState.PENDING
    participants: Set[str] = field(default_factory=set)  # Group IDs
    created_at: datetime = field(default_factory=datetime.now)

    def add_operation(self, op: TransactionOperation) -> None:
        """Add an operation to the transaction."""
        self.operations.append(op)


class TransactionCoordinator:
    """
    Coordinates distributed transactions across Raft groups.

    Implements two-phase commit (2PC) protocol.
    """

    def __init__(self, node: MultiRaftNode):
        self.node = node
        self._active_txns: Dict[str, DistributedTransaction] = {}
        self._txn_locks: Dict[str, asyncio.Lock] = {}

    async def begin_transaction(self) -> DistributedTransaction:
        """Begin a new distributed transaction."""
        txn = DistributedTransaction()
        self._active_txns[txn.txn_id] = txn
        self._txn_locks[txn.txn_id] = asyncio.Lock()
        return txn

    async def add_operation(
        self,
        txn_id: str,
        op_type: str,
        key: bytes,
        value: Any = None,
    ) -> None:
        """Add an operation to a transaction."""
        txn = self._active_txns.get(txn_id)
        if not txn:
            raise ValueError(f"Transaction {txn_id} not found")

        if txn.state != TransactionState.PENDING:
            raise ValueError(f"Transaction {txn_id} is not pending")

        # Determine which group handles this key
        partition_id = self.node._router.get_partition_for_key(key)
        if not partition_id:
            raise ValueError(f"No partition for key")

        partition = self.node._router.get_partition_info(partition_id)
        if not partition:
            raise ValueError(f"Partition not found")

        op = TransactionOperation(
            op_type=op_type,
            key=key,
            value=value,
            group_id=partition.group_id,
        )
        txn.operations.append(op)
        txn.participants.add(partition.group_id)

    async def commit(self, txn_id: str) -> bool:
        """Commit a distributed transaction using 2PC."""
        txn = self._active_txns.get(txn_id)
        if not txn:
            raise ValueError(f"Transaction {txn_id} not found")

        async with self._txn_locks[txn_id]:
            try:
                # Phase 1: Prepare
                txn.state = TransactionState.PREPARING

                prepare_results = await self._prepare_phase(txn)

                if not all(prepare_results.values()):
                    # Abort if any participant failed prepare
                    await self._abort_phase(txn)
                    return False

                txn.state = TransactionState.PREPARED

                # Phase 2: Commit
                txn.state = TransactionState.COMMITTING
                await self._commit_phase(txn)
                txn.state = TransactionState.COMMITTED

                return True

            except Exception as e:
                logger.error(f"Transaction {txn_id} failed: {e}")
                await self._abort_phase(txn)
                return False

            finally:
                # Cleanup
                del self._active_txns[txn_id]
                del self._txn_locks[txn_id]

    async def abort(self, txn_id: str) -> None:
        """Abort a transaction."""
        txn = self._active_txns.get(txn_id)
        if not txn:
            return

        async with self._txn_locks[txn_id]:
            await self._abort_phase(txn)
            del self._active_txns[txn_id]
            del self._txn_locks[txn_id]

    async def _prepare_phase(self, txn: DistributedTransaction) -> Dict[str, bool]:
        """Execute prepare phase on all participants."""
        results = {}

        # Group operations by participant
        ops_by_group: Dict[str, List[TransactionOperation]] = defaultdict(list)
        for op in txn.operations:
            if op.group_id:
                ops_by_group[op.group_id].append(op)

        # Send prepare to each participant
        prepare_tasks = []
        for group_id, ops in ops_by_group.items():
            group = self.node.get_group(group_id)
            if group and group.is_leader():
                task = group.propose({
                    "type": "prepare",
                    "txn_id": txn.txn_id,
                    "operations": [
                        {"type": op.op_type, "key": op.key.decode(), "value": op.value, "group_id": op.group_id}
                        for op in ops
                    ],
                })
                prepare_tasks.append((group_id, task))

        # Wait for all prepares
        for group_id, task in prepare_tasks:
            try:
                result = await task
                results[group_id] = bool(result)
            except Exception as e:
                logger.error(f"Prepare failed for group {group_id}: {e}")
                results[group_id] = False

        return results

    async def _commit_phase(self, txn: DistributedTransaction) -> None:
        """Execute commit phase on all participants."""
        ops_by_group: Dict[str, List[TransactionOperation]] = defaultdict(list)
        for op in txn.operations:
            if op.group_id:
                ops_by_group[op.group_id].append(op)

        commit_tasks = []
        for group_id, ops in ops_by_group.items():
            group = self.node.get_group(group_id)
            if group and group.is_leader():
                task = group.propose({
                    "type": "commit",
                    "txn_id": txn.txn_id,
                    "operations": [
                        {"type": op.op_type, "key": op.key.decode(), "value": op.value, "group_id": op.group_id}
                        for op in ops
                    ],
                })
                commit_tasks.append(task)

        # Wait for all commits
        await asyncio.gather(*commit_tasks, return_exceptions=True)

    async def _abort_phase(self, txn: DistributedTransaction) -> None:
        """Execute abort phase on all participants."""
        txn.state = TransactionState.ABORTING

        abort_tasks = []
        for group_id in txn.participants:
            group = self.node.get_group(group_id)
            if group and group.is_leader():
                task = group.propose({
                    "type": "abort",
                    "txn_id": txn.txn_id,
                })
                abort_tasks.append(task)

        await asyncio.gather(*abort_tasks, return_exceptions=True)
        txn.state = TransactionState.ABORTED


# === Partition Management ===

class PartitionManager:
    """
    Manages partitions across a Multi-Raft cluster.

    Handles:
    - Partition splitting when too large
    - Partition merging when too small
    - Load balancing across nodes
    - Replica placement
    """

    def __init__(
        self,
        node: MultiRaftNode,
        max_partition_size: int = 100 * 1024 * 1024,  # 100MB
        min_partition_size: int = 10 * 1024 * 1024,   # 10MB
        replication_factor: int = 3,
    ):
        self.node = node
        self.max_partition_size = max_partition_size
        self.min_partition_size = min_partition_size
        self.replication_factor = replication_factor

        self._partition_sizes: Dict[str, int] = {}
        self._check_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start partition management."""
        self._check_task = asyncio.create_task(self._check_loop())

    async def stop(self) -> None:
        """Stop partition management."""
        if self._check_task:
            self._check_task.cancel()

    async def _check_loop(self) -> None:
        """Periodically check partition health."""
        while True:
            try:
                await asyncio.sleep(60.0)  # Check every minute

                for partition in self.node._router.get_all_partitions():
                    size = self._partition_sizes.get(partition.partition_id, 0)

                    if size > self.max_partition_size:
                        await self.split_partition(partition.partition_id)

                    elif size < self.min_partition_size:
                        # Find adjacent partition to merge with
                        pass  # Merge logic

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Partition check error: {e}")

    async def split_partition(self, partition_id: str) -> Tuple[str, str]:
        """Split a partition into two."""
        partition = self.node._router.get_partition_info(partition_id)
        if not partition:
            raise ValueError(f"Partition {partition_id} not found")

        # Calculate split key (midpoint)
        start = int.from_bytes(partition.start_key[:8].ljust(8, b'\x00'), 'big')
        end = int.from_bytes(partition.end_key[:8].ljust(8, b'\x00'), 'big')
        mid = (start + end) // 2
        split_key = mid.to_bytes(8, 'big')

        # Create two new partitions
        left_id = f"{partition_id}_L"
        right_id = f"{partition_id}_R"

        left = PartitionInfo(
            partition_id=left_id,
            group_id=f"group_{left_id}",
            start_key=partition.start_key,
            end_key=split_key,
            replicas=partition.replicas.copy(),
            epoch=partition.epoch + 1,
        )

        right = PartitionInfo(
            partition_id=right_id,
            group_id=f"group_{right_id}",
            start_key=split_key,
            end_key=partition.end_key,
            replicas=partition.replicas.copy(),
            epoch=partition.epoch + 1,
        )

        # Update partition state
        partition.state = PartitionState.SPLITTING

        # Add new partitions to router
        self.node._router.add_partition(left)
        self.node._router.add_partition(right)

        # Remove old partition
        self.node._router.remove_partition(partition_id)

        logger.info(f"Split partition {partition_id} into {left_id} and {right_id}")

        return left_id, right_id

    def update_partition_size(self, partition_id: str, size: int) -> None:
        """Update the tracked size of a partition."""
        self._partition_sizes[partition_id] = size
