"""
Comprehensive tests for AION consensus and distributed systems primitives.

Tests cover:
- Raft consensus (leader election, log replication, membership changes)
- SWIM membership protocol
- CRDTs (counters, sets, registers, maps)
- WAL persistence
- Multi-Raft coordination
"""

import asyncio
import os
import tempfile
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
import pytest

# Import all modules
from aion.systems.process.consensus import (
    RaftNode,
    RaftState,
    RaftLog,
    LogEntry,
    Snapshot,
    AppendEntriesRequest,
    AppendEntriesResponse,
    RequestVoteRequest,
    RequestVoteResponse,
    NotLeaderError,
    VectorClock,
    SWIMProtocol,
    SWIMMember,
    SWIMState,
    # New additions
    MembershipManager,
    ClusterConfiguration,
    JointConfiguration,
    ConfigurationState,
    AddServerRequest,
    RemoveServerRequest,
    PipelinedReplicator,
    EnhancedSWIMProtocol,
    GossipMessage,
    GossipMessageType,
)

from aion.systems.process.crdt import (
    GCounter,
    PNCounter,
    LWWRegister,
    MVRegister,
    GSet,
    TwoPSet,
    ORSet,
    LWWMap,
    RGA,
    CRDTManager,
)

from aion.systems.process.wal import (
    WriteAheadLog,
    WALRecord,
    RecordType,
    WALCorruptionError,
    CheckpointManager,
)

from aion.systems.process.multi_raft import (
    MultiRaftNode,
    RaftGroup,
    RaftGroupConfig,
    MultiRaftRouter,
    PartitionInfo,
    PartitionState,
    TransactionCoordinator,
)


# === Raft Consensus Tests ===

class TestRaftLog:
    """Tests for RaftLog."""

    def test_append_and_get(self):
        """Test basic append and get operations."""
        log = RaftLog()

        entry = LogEntry(term=1, index=1, command={"type": "set", "key": "a", "value": 1})
        log.append(entry)

        retrieved = log.get(1)
        assert retrieved is not None
        assert retrieved.term == 1
        assert retrieved.command["key"] == "a"

    def test_append_entries(self):
        """Test appending multiple entries."""
        log = RaftLog()

        entries = [
            LogEntry(term=1, index=1, command={"key": "a"}),
            LogEntry(term=1, index=2, command={"key": "b"}),
            LogEntry(term=2, index=3, command={"key": "c"}),
        ]
        log.append_entries(entries)

        assert log.last_index() == 3
        assert log.last_term() == 2

    def test_truncate_after(self):
        """Test log truncation."""
        log = RaftLog()

        for i in range(1, 6):
            log.append(LogEntry(term=1, index=i, command={"idx": i}))

        log.truncate_after(3)

        assert log.last_index() == 3
        assert log.get(4) is None
        assert log.get(5) is None

    def test_term_at(self):
        """Test getting term at specific index."""
        log = RaftLog()

        log.append(LogEntry(term=1, index=1, command={}))
        log.append(LogEntry(term=2, index=2, command={}))
        log.append(LogEntry(term=2, index=3, command={}))

        assert log.term_at(1) == 1
        assert log.term_at(2) == 2
        assert log.term_at(3) == 2

    def test_snapshot(self):
        """Test snapshot creation and log compaction."""
        log = RaftLog()

        for i in range(1, 11):
            log.append(LogEntry(term=1, index=i, command={"idx": i}))

        snapshot = log.create_snapshot(5, 1, b'{"state": "snapshot"}')

        assert snapshot.last_included_index == 5
        assert log._base_index == 5
        assert log.get(4) is None  # Compacted
        assert log.get(6) is not None  # Still available


class TestRaftNode:
    """Tests for RaftNode."""

    @pytest.fixture
    def nodes(self):
        """Create a 3-node Raft cluster."""
        nodes = {}
        for i in range(3):
            node_id = f"node{i}"
            peers = [f"node{j}" for j in range(3) if j != i]
            nodes[node_id] = RaftNode(
                node_id=node_id,
                peers=peers,
                apply_callback=lambda cmd: cmd,
                election_timeout_range=(0.1, 0.2),
                heartbeat_interval=0.05,
            )
        return nodes

    def test_initial_state(self, nodes):
        """Test nodes start as followers."""
        for node in nodes.values():
            assert node.state == RaftState.FOLLOWER
            assert node.current_term == 0

    @pytest.mark.asyncio
    async def test_handle_request_vote(self, nodes):
        """Test vote request handling."""
        node = nodes["node0"]

        request = RequestVoteRequest(
            term=1,
            candidate_id="node1",
            last_log_index=0,
            last_log_term=0,
        )

        response = await node.handle_request_vote(request)

        assert response.vote_granted
        assert node.voted_for == "node1"

    @pytest.mark.asyncio
    async def test_handle_request_vote_stale_term(self, nodes):
        """Test rejecting votes for stale terms."""
        node = nodes["node0"]
        node.current_term = 5

        request = RequestVoteRequest(
            term=3,
            candidate_id="node1",
            last_log_index=0,
            last_log_term=0,
        )

        response = await node.handle_request_vote(request)

        assert not response.vote_granted
        assert response.term == 5

    @pytest.mark.asyncio
    async def test_handle_append_entries(self, nodes):
        """Test append entries handling."""
        node = nodes["node0"]

        request = AppendEntriesRequest(
            term=1,
            leader_id="node1",
            prev_log_index=0,
            prev_log_term=0,
            entries=[LogEntry(term=1, index=1, command={"key": "a"})],
            leader_commit=0,
        )

        response = await node.handle_append_entries(request)

        assert response.success
        assert node.log.last_index() == 1
        assert node.leader_id == "node1"

    @pytest.mark.asyncio
    async def test_handle_append_entries_conflict(self, nodes):
        """Test conflict detection in append entries."""
        node = nodes["node0"]

        # Add existing entry with different term
        node.log.append(LogEntry(term=1, index=1, command={"old": True}))

        request = AppendEntriesRequest(
            term=2,
            leader_id="node1",
            prev_log_index=1,
            prev_log_term=2,  # Different from existing term 1
            entries=[],
            leader_commit=0,
        )

        response = await node.handle_append_entries(request)

        assert not response.success
        assert response.conflict_index > 0


class TestVectorClock:
    """Tests for VectorClock."""

    def test_tick(self):
        """Test clock increment."""
        vc = VectorClock("node1")
        vc.tick()
        assert vc._clock["node1"] == 1

    def test_update(self):
        """Test clock merge on receive."""
        vc1 = VectorClock("node1")
        vc1._clock = {"node1": 2, "node2": 1}

        vc2 = VectorClock("node2")
        vc2._clock = {"node1": 1, "node2": 3}

        vc1.update(vc2)

        assert vc1._clock["node1"] == 3  # max(2,1) + 1
        assert vc1._clock["node2"] == 3  # max(1,3)

    def test_happens_before(self):
        """Test happens-before relation."""
        vc1 = VectorClock("node1")
        vc1._clock = {"node1": 1, "node2": 2}

        vc2 = VectorClock("node2")
        vc2._clock = {"node1": 1, "node2": 3}

        assert vc1 < vc2
        assert not vc2 < vc1

    def test_concurrent(self):
        """Test concurrent detection."""
        vc1 = VectorClock("node1")
        vc1._clock = {"node1": 2, "node2": 1}

        vc2 = VectorClock("node2")
        vc2._clock = {"node1": 1, "node2": 2}

        assert vc1.concurrent(vc2)
        assert vc2.concurrent(vc1)


class TestMembershipChanges:
    """Tests for joint consensus membership changes."""

    def test_cluster_configuration(self):
        """Test cluster configuration."""
        config = ClusterConfiguration(
            members={"node1", "node2", "node3"},
            learners={"node4"},
        )

        assert config.voting_members() == {"node1", "node2", "node3"}
        assert config.majority_size() == 2

    def test_joint_configuration(self):
        """Test joint configuration majority requirements."""
        old = ClusterConfiguration(members={"node1", "node2", "node3"})
        new = ClusterConfiguration(members={"node2", "node3", "node4"})

        joint = JointConfiguration(old_config=old, new_config=new)

        # Need majority in both
        assert joint.requires_majority_in_both({"node2", "node3"})  # 2/3 old, 2/3 new
        assert not joint.requires_majority_in_both({"node1", "node2"})  # 2/3 old, 1/3 new
        assert not joint.requires_majority_in_both({"node3", "node4"})  # 1/3 old, 2/3 new


# === CRDT Tests ===

class TestGCounter:
    """Tests for GCounter."""

    def test_increment(self):
        """Test counter increment."""
        counter = GCounter("node1")
        counter.increment(5)
        assert counter.value() == 5

    def test_merge(self):
        """Test counter merge."""
        c1 = GCounter("node1")
        c1.increment(3)

        c2 = GCounter("node2")
        c2.increment(5)

        merged = c1.merge(c2)
        assert merged.value() == 8

    def test_merge_same_node(self):
        """Test merge takes max for same node."""
        c1 = GCounter("node1")
        c1._counts = {"node1": 5, "node2": 3}

        c2 = GCounter("node1")
        c2._counts = {"node1": 3, "node2": 7}

        merged = c1.merge(c2)
        assert merged._counts["node1"] == 5
        assert merged._counts["node2"] == 7


class TestPNCounter:
    """Tests for PNCounter."""

    def test_increment_decrement(self):
        """Test increment and decrement."""
        counter = PNCounter("node1")
        counter.increment(10)
        counter.decrement(3)
        assert counter.value() == 7

    def test_merge(self):
        """Test merge of two counters."""
        c1 = PNCounter("node1")
        c1.increment(5)
        c1.decrement(2)

        c2 = PNCounter("node2")
        c2.increment(3)
        c2.decrement(1)

        merged = c1.merge(c2)
        assert merged.value() == 5  # (5+3) - (2+1)


class TestLWWRegister:
    """Tests for LWWRegister."""

    def test_set_get(self):
        """Test basic set and get."""
        reg = LWWRegister("node1")
        reg.set("hello")
        assert reg.get() == "hello"

    def test_merge_last_wins(self):
        """Test that later timestamp wins."""
        r1 = LWWRegister("node1")
        r1.set("first")

        time.sleep(0.01)  # Ensure different timestamp

        r2 = LWWRegister("node2")
        r2.set("second")

        merged = r1.merge(r2)
        assert merged.get() == "second"


class TestORSet:
    """Tests for OR-Set."""

    def test_add_contains(self):
        """Test add and contains."""
        s = ORSet("node1")
        s.add("apple")
        s.add("banana")

        assert s.contains("apple")
        assert s.contains("banana")
        assert not s.contains("cherry")

    def test_remove(self):
        """Test remove operation."""
        s = ORSet("node1")
        s.add("apple")
        assert s.contains("apple")

        s.remove("apple")
        assert not s.contains("apple")

    def test_add_after_remove(self):
        """Test add-remove-add works correctly."""
        s = ORSet("node1")
        s.add("apple")
        s.remove("apple")
        s.add("apple")

        assert s.contains("apple")

    def test_merge_concurrent_add_remove(self):
        """Test merge with concurrent add and remove."""
        s1 = ORSet("node1")
        s1.add("apple")

        s2 = ORSet("node2")
        s2._elements = dict(s1._elements)  # Copy state

        # Concurrent operations
        s1.remove("apple")
        s2.add("apple")

        merged = s1.merge(s2)
        # The new add should be visible
        assert merged.contains("apple")


class TestRGA:
    """Tests for RGA (collaborative text)."""

    def test_insert(self):
        """Test character insertion."""
        rga = RGA("node1")
        rga.insert(0, 'a')
        rga.insert(1, 'b')
        rga.insert(2, 'c')

        assert rga.text() == "abc"

    def test_delete(self):
        """Test character deletion."""
        rga = RGA("node1")
        rga.insert(0, 'a')
        rga.insert(1, 'b')
        rga.insert(2, 'c')

        rga.delete(1)  # Delete 'b'

        assert rga.text() == "ac"

    def test_merge_concurrent_inserts(self):
        """Test merging concurrent insertions."""
        r1 = RGA("node1")
        r1.insert(0, 'a')

        r2 = RGA("node2")
        r2._nodes = dict(r1._nodes)
        r2._order = list(r1._order)

        # Concurrent inserts at same position
        r1.insert(1, 'b')  # node1 inserts 'b' after 'a'
        r2.insert(1, 'c')  # node2 inserts 'c' after 'a'

        merged = r1.merge(r2)
        text = merged.text()

        # Both characters should be present
        assert 'a' in text
        assert 'b' in text
        assert 'c' in text
        assert len(text) == 3


# === WAL Tests ===

class TestWAL:
    """Tests for Write-Ahead Log."""

    @pytest.fixture
    def wal_dir(self):
        """Create temporary directory for WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_append_and_read(self, wal_dir):
        """Test basic append and read."""
        wal = WriteAheadLog(wal_dir, sync_mode="async", batch_size=1)
        wal.open()

        wal.append(term=1, index=1, data=b'{"key": "value"}')
        wal.append(term=1, index=2, data=b'{"key": "value2"}')

        records = list(wal.read(1, 3))
        assert len(records) == 2
        assert records[0].index == 1
        assert records[1].index == 2

        wal.close()

    def test_recovery(self, wal_dir):
        """Test WAL recovery after close."""
        # Write some records
        wal = WriteAheadLog(wal_dir, sync_mode="async", batch_size=1)
        wal.open()

        for i in range(1, 6):
            wal.append(term=1, index=i, data=f'{{"idx": {i}}}'.encode())

        wal.close()

        # Reopen and verify
        wal2 = WriteAheadLog(wal_dir, sync_mode="async", batch_size=1)
        wal2.open()

        assert wal2.get_last_index() == 5
        records = list(wal2.read(1, 6))
        assert len(records) == 5

        wal2.close()

    def test_truncate_before(self, wal_dir):
        """Test log compaction."""
        wal = WriteAheadLog(wal_dir, sync_mode="async", batch_size=1)
        wal.open()

        for i in range(1, 11):
            wal.append(term=1, index=i, data=f'{{"idx": {i}}}'.encode())

        wal.truncate_before(5)

        # Old entries should be gone
        with pytest.raises(ValueError):
            list(wal.read(1))

        wal.close()


# === Multi-Raft Tests ===

class TestMultiRaftRouter:
    """Tests for Multi-Raft router."""

    def test_partition_routing(self):
        """Test key-to-partition routing."""
        router = MultiRaftRouter(hash_slots=1024)

        partition = PartitionInfo(
            partition_id="p1",
            group_id="g1",
            start_key=b'\x00' * 8,
            end_key=b'\xff' * 8,
            replicas={"node1", "node2", "node3"},
        )

        router.add_partition(partition)

        # Any key should route to this partition (it covers full range)
        pid = router.get_partition_for_key(b'test_key')
        assert pid == "p1"

    def test_partition_info(self):
        """Test partition info retrieval."""
        router = MultiRaftRouter()

        partition = PartitionInfo(
            partition_id="p1",
            group_id="g1",
            start_key=b'\x00',
            end_key=b'\xff',
        )

        router.add_partition(partition)

        info = router.get_partition_info("p1")
        assert info is not None
        assert info.group_id == "g1"


# === SWIM Protocol Tests ===

class TestSWIMProtocol:
    """Tests for SWIM membership protocol."""

    def test_add_member(self):
        """Test adding members."""
        swim = SWIMProtocol("node1", "127.0.0.1:8001")
        swim.add_member("node2", "127.0.0.1:8002")
        swim.add_member("node3", "127.0.0.1:8003")

        assert len(swim._members) == 2
        assert "node2" in swim._members
        assert "node3" in swim._members

    def test_get_alive_members(self):
        """Test getting alive members."""
        swim = SWIMProtocol("node1", "127.0.0.1:8001")
        swim.add_member("node2", "127.0.0.1:8002")
        swim.add_member("node3", "127.0.0.1:8003")

        alive = swim.get_alive_members()
        assert len(alive) == 2

    def test_refute_increments_incarnation(self):
        """Test that refute increments incarnation."""
        swim = SWIMProtocol("node1", "127.0.0.1:8001")

        old_incarnation = swim.incarnation
        swim.refute()

        assert swim.incarnation == old_incarnation + 1


class TestEnhancedSWIM:
    """Tests for enhanced SWIM with gossip."""

    def test_queue_gossip(self):
        """Test gossip message queueing."""
        swim = EnhancedSWIMProtocol("node1", "127.0.0.1:8001")

        msg = GossipMessage(
            msg_type=GossipMessageType.ALIVE,
            member_id="node2",
            incarnation=1,
            address="127.0.0.1:8002",
        )

        swim.queue_gossip(msg)
        assert len(swim._gossip_buffer) == 1

    def test_get_gossip_payload(self):
        """Test getting gossip payload for piggybacking."""
        swim = EnhancedSWIMProtocol("node1", "127.0.0.1:8001")
        swim.add_member("node2", "127.0.0.1:8002")
        swim.add_member("node3", "127.0.0.1:8003")

        msg = GossipMessage(
            msg_type=GossipMessageType.ALIVE,
            member_id="node2",
            incarnation=1,
        )
        swim.queue_gossip(msg)

        payload = swim.get_gossip_payload("node3")
        assert len(payload) >= 1

    def test_process_gossip_join(self):
        """Test processing join gossip."""
        swim = EnhancedSWIMProtocol("node1", "127.0.0.1:8001")

        join_msg = [{
            "type": "JOIN",
            "member_id": "node2",
            "incarnation": 0,
            "address": "127.0.0.1:8002",
        }]

        swim.process_gossip_payload(join_msg)

        assert "node2" in swim._members
        assert swim._members["node2"].state == SWIMState.ALIVE


# === Integration Tests ===

class TestCRDTManager:
    """Tests for CRDT manager."""

    def test_create_and_sync(self):
        """Test creating CRDTs and syncing."""
        mgr1 = CRDTManager("node1")
        mgr2 = CRDTManager("node2")

        # Create counters
        c1 = mgr1.create_counter("visits")
        c1.increment(5)

        c2 = mgr2.create_counter("visits")
        c2.increment(3)

        # Sync from mgr1 to mgr2
        state = mgr1.get_state("visits")
        mgr2.merge_remote("visits", state)

        # mgr2 should now have combined count
        assert mgr2.get("visits").value() == 8


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
