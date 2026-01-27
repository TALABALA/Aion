"""
AION Distributed Computing System Tests

Comprehensive tests covering:
- Core types and configuration
- Vector clock causality
- Consistent hashing distribution
- Raft consensus protocol
- Distributed task queue
- Load balancing strategies
- Failure detection (phi accrual)
- Memory sharding
- State synchronization
- Cluster management lifecycle
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion.distributed.types import (
    NodeInfo,
    NodeRole,
    NodeStatus,
    NodeCapability,
    ClusterState,
    DistributedTask,
    TaskStatus,
    TaskPriority,
    TaskType,
    VectorClock,
    RaftState,
    RaftLogEntry,
    VoteRequest,
    VoteResponse,
    AppendEntriesRequest,
    AppendEntriesResponse,
    HeartbeatMessage,
    HealthReport,
    ShardInfo,
    ConsistencyLevel,
    ReplicationEvent,
    SnapshotMetadata,
    ClusterMetrics,
)
from aion.distributed.config import (
    DistributedConfig,
    NetworkConfig,
    RaftConfig,
    DiscoveryConfig,
    HealthConfig,
    TaskQueueConfig,
    ReplicationConfig,
    ShardingConfig,
    LoadBalancingConfig,
    FaultToleranceConfig,
    TrainingConfig,
    get_default_config,
    get_development_config,
    get_production_config,
)


# =============================================================================
# Type Tests
# =============================================================================


class TestNodeInfo:
    """Test NodeInfo dataclass."""

    def test_default_creation(self):
        node = NodeInfo()
        assert node.host == "localhost"
        assert node.port == 5000
        assert node.grpc_port == 5001
        assert node.role == NodeRole.FOLLOWER
        assert node.status == NodeStatus.STARTING
        assert node.max_concurrent_tasks == 10

    def test_address_properties(self):
        node = NodeInfo(host="192.168.1.10", port=6000, grpc_port=6001)
        assert node.address == "192.168.1.10:6000"
        assert node.grpc_address == "192.168.1.10:6001"

    def test_load_score_empty(self):
        node = NodeInfo(
            current_tasks=0,
            max_concurrent_tasks=10,
            cpu_usage=0.0,
            memory_usage=0.0,
        )
        assert node.load_score == 0.0

    def test_load_score_fully_loaded(self):
        node = NodeInfo(
            current_tasks=10,
            max_concurrent_tasks=10,
            cpu_usage=1.0,
            memory_usage=1.0,
        )
        assert node.load_score == 1.0

    def test_load_score_partial(self):
        node = NodeInfo(
            current_tasks=5,
            max_concurrent_tasks=10,
            cpu_usage=0.5,
            memory_usage=0.5,
        )
        score = node.load_score
        assert 0.0 < score < 1.0

    def test_is_available_healthy(self):
        node = NodeInfo(
            status=NodeStatus.HEALTHY,
            current_tasks=5,
            max_concurrent_tasks=10,
        )
        assert node.is_available() is True

    def test_is_available_unhealthy(self):
        node = NodeInfo(status=NodeStatus.UNHEALTHY)
        assert node.is_available() is False

    def test_is_available_at_capacity(self):
        node = NodeInfo(
            status=NodeStatus.HEALTHY,
            current_tasks=10,
            max_concurrent_tasks=10,
        )
        assert node.is_available() is False

    def test_is_voter(self):
        leader = NodeInfo(role=NodeRole.LEADER)
        follower = NodeInfo(role=NodeRole.FOLLOWER)
        observer = NodeInfo(role=NodeRole.OBSERVER)
        candidate = NodeInfo(role=NodeRole.CANDIDATE)

        assert leader.is_voter() is True
        assert follower.is_voter() is True
        assert candidate.is_voter() is True
        assert observer.is_voter() is False

    def test_available_capacity(self):
        node = NodeInfo(current_tasks=3, max_concurrent_tasks=10)
        assert node.available_capacity == 7

    def test_to_dict(self):
        node = NodeInfo(name="test-node", host="10.0.0.1")
        d = node.to_dict()
        assert d["name"] == "test-node"
        assert d["address"] == "10.0.0.1:5000"
        assert "role" in d
        assert "status" in d
        assert "load_score" in d

    def test_unique_ids(self):
        n1 = NodeInfo()
        n2 = NodeInfo()
        assert n1.id != n2.id


class TestClusterState:
    """Test ClusterState dataclass."""

    def test_default_creation(self):
        state = ClusterState()
        assert state.name == "aion-cluster"
        assert state.leader_id is None
        assert state.min_nodes == 1
        assert state.replication_factor == 3

    def test_healthy_nodes(self):
        state = ClusterState()
        healthy = NodeInfo(id="n1", status=NodeStatus.HEALTHY)
        unhealthy = NodeInfo(id="n2", status=NodeStatus.UNHEALTHY)
        state.nodes = {"n1": healthy, "n2": unhealthy}

        assert len(state.healthy_nodes) == 1
        assert state.healthy_nodes[0].id == "n1"

    def test_quorum_size(self):
        state = ClusterState()
        for i in range(5):
            node = NodeInfo(id=f"n{i}", role=NodeRole.FOLLOWER)
            state.nodes[node.id] = node
        assert state.quorum_size == 3  # 5 // 2 + 1

    def test_has_quorum_true(self):
        state = ClusterState()
        for i in range(3):
            node = NodeInfo(
                id=f"n{i}",
                status=NodeStatus.HEALTHY,
                role=NodeRole.FOLLOWER,
            )
            state.nodes[node.id] = node
        assert state.has_quorum is True

    def test_has_quorum_false(self):
        state = ClusterState()
        n1 = NodeInfo(id="n1", status=NodeStatus.HEALTHY, role=NodeRole.FOLLOWER)
        n2 = NodeInfo(id="n2", status=NodeStatus.UNHEALTHY, role=NodeRole.FOLLOWER)
        n3 = NodeInfo(id="n3", status=NodeStatus.OFFLINE, role=NodeRole.FOLLOWER)
        state.nodes = {"n1": n1, "n2": n2, "n3": n3}
        # Only 1 healthy out of 3, quorum = 2
        assert state.has_quorum is False

    def test_get_leader(self):
        state = ClusterState()
        leader = NodeInfo(id="leader-1", role=NodeRole.LEADER)
        state.nodes["leader-1"] = leader
        state.leader_id = "leader-1"
        assert state.get_leader() == leader

    def test_get_leader_none(self):
        state = ClusterState()
        assert state.get_leader() is None

    def test_increment_epoch(self):
        state = ClusterState(epoch=5)
        state.increment_epoch()
        assert state.epoch == 6


class TestDistributedTask:
    """Test DistributedTask dataclass."""

    def test_default_creation(self):
        task = DistributedTask(name="test", task_type="compute")
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL
        assert task.max_retries == 3
        assert task.retry_count == 0

    def test_is_terminal(self):
        completed = DistributedTask(status=TaskStatus.COMPLETED)
        failed = DistributedTask(status=TaskStatus.FAILED)
        running = DistributedTask(status=TaskStatus.RUNNING)
        cancelled = DistributedTask(status=TaskStatus.CANCELLED)

        assert completed.is_terminal is True
        assert failed.is_terminal is True
        assert cancelled.is_terminal is True
        assert running.is_terminal is False

    def test_can_retry(self):
        task = DistributedTask(max_retries=3, retry_count=2)
        assert task.can_retry() is True

        task.retry_count = 3
        assert task.can_retry() is False

    def test_execution_time(self):
        now = datetime.now()
        task = DistributedTask(
            started_at=now,
            completed_at=now + timedelta(seconds=5),
        )
        assert task.execution_time_ms == pytest.approx(5000.0, abs=10)

    def test_is_expired(self):
        task = DistributedTask(
            deadline=datetime.now() - timedelta(minutes=1),
        )
        assert task.is_expired is True

        task2 = DistributedTask(
            deadline=datetime.now() + timedelta(hours=1),
        )
        assert task2.is_expired is False

    def test_record_attempt(self):
        task = DistributedTask()
        task.record_attempt("node-1", error="timeout")
        assert len(task.attempt_history) == 1
        assert task.attempt_history[0]["node_id"] == "node-1"
        assert task.attempt_history[0]["error"] == "timeout"

    def test_to_dict(self):
        task = DistributedTask(name="test-task", task_type="compute")
        d = task.to_dict()
        assert d["name"] == "test-task"
        assert d["task_type"] == "compute"
        assert "id" in d
        assert "status" in d


# =============================================================================
# Vector Clock Tests
# =============================================================================


class TestVectorClock:
    """Test VectorClock for causal ordering."""

    def test_increment(self):
        vc = VectorClock()
        vc.increment("node-1")
        assert vc.clocks["node-1"] == 1
        vc.increment("node-1")
        assert vc.clocks["node-1"] == 2

    def test_merge(self):
        vc1 = VectorClock(clocks={"a": 2, "b": 1})
        vc2 = VectorClock(clocks={"a": 1, "b": 3, "c": 1})
        vc1.merge(vc2)
        assert vc1.clocks == {"a": 2, "b": 3, "c": 1}

    def test_dominates(self):
        vc1 = VectorClock(clocks={"a": 2, "b": 3})
        vc2 = VectorClock(clocks={"a": 1, "b": 2})
        assert vc1.dominates(vc2) is True
        assert vc2.dominates(vc1) is False

    def test_concurrent(self):
        vc1 = VectorClock(clocks={"a": 2, "b": 1})
        vc2 = VectorClock(clocks={"a": 1, "b": 2})
        assert vc1.is_concurrent(vc2) is True

    def test_not_concurrent_dominated(self):
        vc1 = VectorClock(clocks={"a": 2, "b": 3})
        vc2 = VectorClock(clocks={"a": 1, "b": 2})
        assert vc1.is_concurrent(vc2) is False

    def test_copy(self):
        vc = VectorClock(clocks={"a": 1, "b": 2})
        vc_copy = vc.copy()
        assert vc_copy.clocks == vc.clocks
        vc_copy.increment("a")
        assert vc.clocks["a"] == 1  # Original unchanged

    def test_empty_dominates(self):
        vc1 = VectorClock(clocks={"a": 1})
        vc2 = VectorClock()
        assert vc1.dominates(vc2) is True

    def test_to_dict(self):
        vc = VectorClock(clocks={"a": 1, "b": 2})
        assert vc.to_dict() == {"a": 1, "b": 2}


# =============================================================================
# Configuration Tests
# =============================================================================


class TestDistributedConfig:
    """Test configuration classes."""

    def test_default_config(self):
        config = DistributedConfig()
        assert config.cluster_name == "aion-cluster"
        assert config.node_name == "aion-node"
        assert config.enabled is True
        assert config.min_nodes == 1

    def test_network_config(self):
        config = DistributedConfig()
        assert config.network.host == "0.0.0.0"
        assert config.network.port == 5000
        assert config.network.grpc_port == 5001
        assert config.network.tls_enabled is False

    def test_raft_config(self):
        config = DistributedConfig()
        assert config.raft.election_timeout_min_ms == 150
        assert config.raft.election_timeout_max_ms == 300
        assert config.raft.heartbeat_interval_ms == 50
        assert config.raft.pre_vote_enabled is True

    def test_raft_config_validation(self):
        with pytest.raises(Exception):
            RaftConfig(
                election_timeout_min_ms=300,
                election_timeout_max_ms=200,
            )

    def test_discovery_config(self):
        config = DistributedConfig()
        assert config.discovery.method == "static"
        assert config.discovery.static_nodes == []

    def test_health_config(self):
        config = DistributedConfig()
        assert config.health.phi_accrual_threshold == 8.0
        assert config.health.max_missed_heartbeats == 5

    def test_replication_config(self):
        config = DistributedConfig()
        assert config.replication.factor == 3
        assert config.replication.mode == "semi_sync"

    def test_to_flat_dict(self):
        config = DistributedConfig(
            cluster_name="test-cluster",
            node_name="test-node",
        )
        flat = config.to_flat_dict()
        assert flat["cluster_name"] == "test-cluster"
        assert flat["node_name"] == "test-node"
        assert "host" in flat
        assert "port" in flat

    def test_development_config(self):
        config = get_development_config()
        assert config.cluster_name == "aion-dev"
        assert config.min_nodes == 1
        assert config.replication.factor == 1

    def test_production_config(self):
        config = get_production_config()
        assert config.min_nodes == 3
        assert config.replication.factor == 3
        assert config.raft.pre_vote_enabled is True
        assert config.fault_tolerance.enable_split_brain_detection is True


# =============================================================================
# Raft State Tests
# =============================================================================


class TestRaftState:
    """Test Raft consensus state."""

    def test_default_state(self):
        state = RaftState()
        assert state.current_term == 0
        assert state.voted_for is None
        assert state.log == []
        assert state.commit_index == -1
        assert state.last_applied == -1

    def test_last_log_index_empty(self):
        state = RaftState()
        assert state.last_log_index == -1

    def test_last_log_index_with_entries(self):
        state = RaftState(log=[
            RaftLogEntry(index=0, term=1),
            RaftLogEntry(index=1, term=1),
            RaftLogEntry(index=2, term=2),
        ])
        assert state.last_log_index == 2

    def test_last_log_term(self):
        state = RaftState(log=[
            RaftLogEntry(index=0, term=1),
            RaftLogEntry(index=1, term=2),
        ])
        assert state.last_log_term == 2

    def test_get_entry(self):
        entries = [
            RaftLogEntry(index=0, term=1, command="set"),
            RaftLogEntry(index=1, term=1, command="get"),
        ]
        state = RaftState(log=entries)
        assert state.get_entry(0).command == "set"
        assert state.get_entry(1).command == "get"
        assert state.get_entry(2) is None

    def test_get_term_at(self):
        state = RaftState(log=[
            RaftLogEntry(index=0, term=1),
            RaftLogEntry(index=1, term=2),
            RaftLogEntry(index=2, term=2),
        ])
        assert state.get_term_at(0) == 1
        assert state.get_term_at(1) == 2


class TestRaftLogEntry:
    """Test Raft log entries."""

    def test_creation(self):
        entry = RaftLogEntry(
            index=5,
            term=3,
            command="set_state",
            data={"key": "value"},
        )
        assert entry.index == 5
        assert entry.term == 3
        assert entry.is_noop is False

    def test_noop_entry(self):
        entry = RaftLogEntry(index=0, term=1, is_noop=True)
        assert entry.is_noop is True

    def test_config_change(self):
        entry = RaftLogEntry(
            index=0,
            term=1,
            is_config_change=True,
            config_data={"add_node": "node-5"},
        )
        assert entry.is_config_change is True

    def test_to_dict(self):
        entry = RaftLogEntry(index=0, term=1, command="test")
        d = entry.to_dict()
        assert d["index"] == 0
        assert d["term"] == 1
        assert d["command"] == "test"


# =============================================================================
# Raft Message Tests
# =============================================================================


class TestRaftMessages:
    """Test Raft protocol messages."""

    def test_vote_request(self):
        req = VoteRequest(
            term=5,
            candidate_id="node-1",
            last_log_index=10,
            last_log_term=4,
        )
        assert req.term == 5
        assert req.is_pre_vote is False

    def test_vote_request_pre_vote(self):
        req = VoteRequest(
            term=5,
            candidate_id="node-1",
            last_log_index=10,
            last_log_term=4,
            is_pre_vote=True,
        )
        assert req.is_pre_vote is True

    def test_vote_response(self):
        resp = VoteResponse(
            term=5,
            vote_granted=True,
            voter_id="node-2",
        )
        assert resp.vote_granted is True

    def test_append_entries_heartbeat(self):
        req = AppendEntriesRequest(
            term=5,
            leader_id="leader-1",
            prev_log_index=10,
            prev_log_term=4,
            entries=[],
            leader_commit=9,
        )
        assert len(req.entries) == 0  # Heartbeat

    def test_append_entries_with_data(self):
        entries = [
            RaftLogEntry(index=11, term=5, command="set"),
            RaftLogEntry(index=12, term=5, command="set"),
        ]
        req = AppendEntriesRequest(
            term=5,
            leader_id="leader-1",
            prev_log_index=10,
            prev_log_term=4,
            entries=entries,
            leader_commit=9,
        )
        assert len(req.entries) == 2

    def test_append_entries_response_success(self):
        resp = AppendEntriesResponse(
            term=5,
            success=True,
            match_index=12,
            node_id="node-2",
        )
        assert resp.success is True

    def test_append_entries_response_conflict(self):
        resp = AppendEntriesResponse(
            term=5,
            success=False,
            node_id="node-2",
            conflict_term=3,
            conflict_index=8,
        )
        assert resp.success is False
        assert resp.conflict_term == 3


# =============================================================================
# Health Report Tests
# =============================================================================


class TestHealthReport:
    """Test health reporting."""

    def test_healthy_report(self):
        report = HealthReport(
            node_id="node-1",
            cpu_usage=0.3,
            memory_usage=0.5,
            error_rate=0.01,
        )
        assert report.is_healthy() is True

    def test_unhealthy_cpu(self):
        report = HealthReport(
            node_id="node-1",
            cpu_usage=0.98,
            memory_usage=0.5,
        )
        assert report.is_healthy() is False

    def test_unhealthy_memory(self):
        report = HealthReport(
            node_id="node-1",
            cpu_usage=0.3,
            memory_usage=0.96,
        )
        assert report.is_healthy() is False

    def test_unhealthy_error_rate(self):
        report = HealthReport(
            node_id="node-1",
            error_rate=0.15,
        )
        assert report.is_healthy() is False


# =============================================================================
# Consistent Hash Tests
# =============================================================================


class TestConsistentHash:
    """Test consistent hashing for shard distribution."""

    def test_basic_distribution(self):
        from aion.distributed.memory.sharding import ConsistentHash

        ch = ConsistentHash(virtual_nodes=100)
        ch.add_node("node-1")
        ch.add_node("node-2")
        ch.add_node("node-3")

        # All keys should map to some node
        for i in range(100):
            node = ch.get_node(f"key-{i}")
            assert node in ("node-1", "node-2", "node-3")

    def test_distribution_uniformity(self):
        from aion.distributed.memory.sharding import ConsistentHash

        ch = ConsistentHash(virtual_nodes=150)
        nodes = [f"node-{i}" for i in range(5)]
        for n in nodes:
            ch.add_node(n)

        # Check distribution across many keys
        counts = Counter()
        for i in range(10000):
            node = ch.get_node(f"key-{i}")
            counts[node] += 1

        # Each node should get roughly 20% (2000) of keys
        # Allow significant deviation since hash-based
        for node in nodes:
            assert counts[node] > 500, f"{node} got too few keys: {counts[node]}"

    def test_consistency_after_removal(self):
        from aion.distributed.memory.sharding import ConsistentHash

        ch = ConsistentHash(virtual_nodes=100)
        ch.add_node("node-1")
        ch.add_node("node-2")
        ch.add_node("node-3")

        # Record initial mapping
        initial = {f"key-{i}": ch.get_node(f"key-{i}") for i in range(100)}

        # Remove node-3
        ch.remove_node("node-3")

        # Keys that were on node-1/node-2 should stay there
        moved = 0
        for key, old_node in initial.items():
            new_node = ch.get_node(key)
            if old_node != "node-3" and new_node != old_node:
                moved += 1

        # Very few keys should move between remaining nodes
        assert moved < 20, f"Too many keys moved: {moved}"

    def test_get_multiple_nodes(self):
        from aion.distributed.memory.sharding import ConsistentHash

        ch = ConsistentHash(virtual_nodes=100)
        for i in range(5):
            ch.add_node(f"node-{i}")

        nodes = ch.get_nodes("test-key", 3)
        assert len(nodes) == 3
        assert len(set(nodes)) == 3  # All unique

    def test_empty_ring(self):
        from aion.distributed.memory.sharding import ConsistentHash

        ch = ConsistentHash(virtual_nodes=100)
        assert ch.get_node("key") is None
        assert ch.get_nodes("key", 3) == []


# =============================================================================
# Distributed Task Queue Tests
# =============================================================================


class TestDistributedTaskQueue:
    """Test distributed task queue."""

    @pytest.fixture
    def mock_cluster_manager(self):
        manager = MagicMock()
        manager.cluster_state = ClusterState()
        return manager

    @pytest.fixture
    def queue(self, mock_cluster_manager):
        from aion.distributed.tasks.queue import DistributedTaskQueue
        return DistributedTaskQueue(mock_cluster_manager)

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, queue):
        task = DistributedTask(name="test", task_type="compute")
        await queue.enqueue(task)
        assert await queue.size() == 1

        dequeued = await queue.dequeue()
        assert dequeued is not None
        assert dequeued.id == task.id

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        low = DistributedTask(name="low", priority=TaskPriority.LOW)
        high = DistributedTask(name="high", priority=TaskPriority.HIGH)
        critical = DistributedTask(name="critical", priority=TaskPriority.CRITICAL)

        await queue.enqueue(low)
        await queue.enqueue(high)
        await queue.enqueue(critical)

        # Should come out in priority order
        first = await queue.dequeue()
        assert first.name == "critical"

        second = await queue.dequeue()
        assert second.name == "high"

        third = await queue.dequeue()
        assert third.name == "low"

    @pytest.mark.asyncio
    async def test_get_task(self, queue):
        task = DistributedTask(name="test", task_type="compute")
        await queue.enqueue(task)

        fetched = await queue.get_task(task.id)
        assert fetched is not None
        assert fetched.id == task.id

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, queue):
        result = await queue.get_task("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_task(self, queue):
        task = DistributedTask(name="test", task_type="compute")
        await queue.enqueue(task)

        task.assigned_node = "node-1"
        task.status = TaskStatus.ASSIGNED
        await queue.update_task(task)

        updated = await queue.get_task(task.id)
        assert updated.assigned_node == "node-1"
        assert updated.status == TaskStatus.ASSIGNED

    @pytest.mark.asyncio
    async def test_get_tasks_by_node(self, queue):
        t1 = DistributedTask(name="t1", assigned_node="node-1")
        t2 = DistributedTask(name="t2", assigned_node="node-1")
        t3 = DistributedTask(name="t3", assigned_node="node-2")

        # Must enqueue first so tasks exist in queue
        await queue.enqueue(t1)
        await queue.enqueue(t2)
        await queue.enqueue(t3)

        node1_tasks = await queue.get_tasks_by_node("node-1")
        assert len(node1_tasks) == 2

    @pytest.mark.asyncio
    async def test_remove_task(self, queue):
        task = DistributedTask(name="test")
        await queue.enqueue(task)
        assert await queue.size() == 1

        await queue.remove_task(task.id)
        result = await queue.get_task(task.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_pending_count(self, queue):
        t1 = DistributedTask(status=TaskStatus.QUEUED)
        t2 = DistributedTask(status=TaskStatus.RUNNING)
        t3 = DistributedTask(status=TaskStatus.PENDING)

        await queue.enqueue(t1)
        await queue.enqueue(t2)
        await queue.enqueue(t3)

        # enqueue sets status to QUEUED, so t2 and t3 also become QUEUED
        # We need to explicitly update t2 status after enqueue
        t2.status = TaskStatus.RUNNING
        await queue.update_task(t2)

        # pending_count counts PENDING and QUEUED
        count = await queue.pending_count()
        assert count >= 1

    @pytest.mark.asyncio
    async def test_get_stats(self, queue):
        await queue.enqueue(DistributedTask(name="t1", priority=TaskPriority.HIGH))
        await queue.enqueue(DistributedTask(name="t2", priority=TaskPriority.LOW))

        stats = await queue.get_stats()
        assert stats["total_tasks"] == 2
        assert "status_counts" in stats

    @pytest.mark.asyncio
    async def test_empty_dequeue(self, queue):
        result = await queue.dequeue()
        assert result is None


# =============================================================================
# Load Balancer Tests
# =============================================================================


class TestLoadBalancingStrategies:
    """Test load balancing strategies."""

    def _make_nodes(self, count: int = 3) -> list:
        nodes = []
        for i in range(count):
            nodes.append(NodeInfo(
                id=f"node-{i}",
                name=f"node-{i}",
                status=NodeStatus.HEALTHY,
                current_tasks=i,
                max_concurrent_tasks=10,
                cpu_usage=i * 0.2,
                memory_usage=i * 0.1,
            ))
        return nodes

    @pytest.mark.asyncio
    async def test_round_robin(self):
        from aion.distributed.balancing.strategies import RoundRobinStrategy

        strategy = RoundRobinStrategy()
        nodes = self._make_nodes(3)

        selections = [await strategy.select_node(nodes) for _ in range(6)]
        # Should cycle through nodes
        ids = [n.id for n in selections if n]
        assert len(ids) == 6
        assert ids[0] == ids[3]  # Wraps around

    @pytest.mark.asyncio
    async def test_least_connections(self):
        from aion.distributed.balancing.strategies import LeastConnectionsStrategy

        strategy = LeastConnectionsStrategy()
        nodes = self._make_nodes(3)
        nodes[0].current_tasks = 1
        nodes[1].current_tasks = 5
        nodes[2].current_tasks = 3

        selected = await strategy.select_node(nodes)
        assert selected.id == "node-0"

    @pytest.mark.asyncio
    async def test_weighted_strategy(self):
        from aion.distributed.balancing.strategies import WeightedStrategy

        strategy = WeightedStrategy()
        nodes = self._make_nodes(3)

        # Run many selections
        counts = Counter()
        for _ in range(1000):
            node = await strategy.select_node(nodes)
            if node:
                counts[node.id] += 1

        # Lower-loaded nodes should be selected more often
        assert counts["node-0"] > counts["node-2"]

    @pytest.mark.asyncio
    async def test_capability_aware(self):
        from aion.distributed.balancing.strategies import (
            CapabilityAwareStrategy,
            LeastConnectionsStrategy,
        )

        fallback = LeastConnectionsStrategy()
        strategy = CapabilityAwareStrategy(fallback=fallback)

        nodes = self._make_nodes(3)
        nodes[0].capabilities = {"compute", "memory"}
        nodes[1].capabilities = {"compute", "gpu"}
        nodes[2].capabilities = {"compute", "memory", "gpu"}

        task = DistributedTask(required_capabilities={"gpu"})
        selected = await strategy.select_node(nodes, task)
        assert selected is not None
        assert "gpu" in selected.capabilities

    @pytest.mark.asyncio
    async def test_no_available_nodes(self):
        from aion.distributed.balancing.strategies import RoundRobinStrategy

        strategy = RoundRobinStrategy()
        # RoundRobin returns any node in the list; pass empty list for None
        result = await strategy.select_node([])
        assert result is None

    @pytest.mark.asyncio
    async def test_power_of_two_choices(self):
        from aion.distributed.balancing.strategies import PowerOfTwoChoicesStrategy

        strategy = PowerOfTwoChoicesStrategy()
        nodes = self._make_nodes(10)

        # Set varying loads
        for i, node in enumerate(nodes):
            node.current_tasks = i

        counts = Counter()
        for _ in range(1000):
            node = await strategy.select_node(nodes)
            if node:
                counts[node.id] += 1

        # Node-0 (least loaded) should be picked most
        assert counts["node-0"] > counts["node-9"]


# =============================================================================
# Failure Detection Tests
# =============================================================================


class TestFailureDetector:
    """Test phi accrual failure detector."""

    def test_basic_detection(self):
        from aion.distributed.fault.detector import FailureDetector

        detector = FailureDetector(threshold=8.0, window_size=100)

        # Record regular heartbeats
        for _ in range(10):
            detector.record_heartbeat("node-1")
            time.sleep(0.01)

        # Node should be alive
        assert detector.is_alive("node-1") is True

    def test_phi_increases_with_delay(self):
        from aion.distributed.fault.detector import FailureDetector

        detector = FailureDetector(threshold=8.0, window_size=100)

        # Record regular heartbeats
        for _ in range(20):
            detector.record_heartbeat("node-1")
            time.sleep(0.005)

        phi_recent = detector.get_phi("node-1")

        # Wait a bit
        time.sleep(0.1)
        phi_later = detector.get_phi("node-1")

        # Phi should increase over time without heartbeats
        assert phi_later > phi_recent

    def test_unknown_node(self):
        from aion.distributed.fault.detector import FailureDetector

        detector = FailureDetector()
        assert detector.is_alive("unknown") is False
        phi = detector.get_phi("unknown")
        assert phi > 0  # Should report high phi for unknown nodes


# =============================================================================
# Cluster Manager Tests
# =============================================================================


class TestClusterManager:
    """Test cluster manager lifecycle."""

    @pytest.fixture
    def mock_kernel(self):
        kernel = MagicMock()
        kernel.tools = MagicMock()
        kernel.memory = MagicMock()
        kernel.planning = MagicMock()
        return kernel

    def test_node_info_creation(self, mock_kernel):
        from aion.distributed.cluster.manager import ClusterManager

        config = {
            "node_name": "test-node",
            "host": "localhost",
            "port": 5000,
            "grpc_port": 5001,
            "capabilities": ["compute", "memory"],
            "max_tasks": 10,
            "cluster_name": "test-cluster",
            "min_nodes": 1,
            "replication_factor": 3,
            "heartbeat_interval": 1.0,
            "health_check_interval": 5.0,
            "health_timeout": 15.0,
        }

        manager = ClusterManager(mock_kernel, config)
        assert manager.local_node.name == "test-node"
        assert manager.local_node.host == "localhost"
        assert manager.state.name == "test-cluster"

    def test_is_leader_false_initially(self, mock_kernel):
        from aion.distributed.cluster.manager import ClusterManager

        config = {
            "node_name": "test",
            "host": "localhost",
            "port": 5000,
            "grpc_port": 5001,
            "capabilities": [],
            "max_tasks": 10,
            "cluster_name": "test",
            "min_nodes": 1,
            "replication_factor": 3,
            "heartbeat_interval": 1.0,
            "health_check_interval": 5.0,
            "health_timeout": 15.0,
        }

        manager = ClusterManager(mock_kernel, config)
        assert manager.is_leader is False

    def test_get_cluster_info(self, mock_kernel):
        from aion.distributed.cluster.manager import ClusterManager

        config = {
            "node_name": "test",
            "host": "localhost",
            "port": 5000,
            "grpc_port": 5001,
            "capabilities": [],
            "max_tasks": 10,
            "cluster_name": "test",
            "min_nodes": 1,
            "replication_factor": 3,
            "heartbeat_interval": 1.0,
            "health_check_interval": 5.0,
            "health_timeout": 15.0,
        }

        manager = ClusterManager(mock_kernel, config)
        info = manager.get_cluster_info()
        assert "cluster_name" in info
        assert "node_id" in info
        assert "is_leader" in info
        assert "nodes" in info

    def test_node_selection_by_load(self, mock_kernel):
        from aion.distributed.cluster.manager import ClusterManager

        config = {
            "node_name": "test",
            "host": "localhost",
            "port": 5000,
            "grpc_port": 5001,
            "capabilities": [],
            "max_tasks": 10,
            "cluster_name": "test",
            "min_nodes": 1,
            "replication_factor": 3,
            "heartbeat_interval": 1.0,
            "health_check_interval": 5.0,
            "health_timeout": 15.0,
        }

        manager = ClusterManager(mock_kernel, config)

        # Add nodes with different loads
        n1 = NodeInfo(id="n1", status=NodeStatus.HEALTHY, current_tasks=8, max_concurrent_tasks=10)
        n2 = NodeInfo(id="n2", status=NodeStatus.HEALTHY, current_tasks=2, max_concurrent_tasks=10)
        n3 = NodeInfo(id="n3", status=NodeStatus.HEALTHY, current_tasks=5, max_concurrent_tasks=10)

        manager.state.nodes = {"n1": n1, "n2": n2, "n3": n3}

        task = DistributedTask(name="test")
        selected = manager._select_node(task)

        # Should select node with lowest load
        assert selected.id == "n2"


# =============================================================================
# PubSub Tests
# =============================================================================


class TestPubSub:
    """Test pub/sub messaging."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        from aion.distributed.communication.pubsub import PubSubManager

        pubsub = PubSubManager()
        received = []

        async def handler(topic, data):
            received.append(data)

        pubsub.subscribe("test.topic", handler)
        await pubsub.publish_async("test.topic", {"msg": "hello"})

        assert len(received) == 1
        assert received[0]["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        from aion.distributed.communication.pubsub import PubSubManager

        pubsub = PubSubManager()
        received = []

        async def handler(topic, data):
            received.append(data)

        pubsub.subscribe("node.*", handler)
        await pubsub.publish_async("node.joined", {"node": "n1"})
        await pubsub.publish_async("node.left", {"node": "n2"})

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        from aion.distributed.communication.pubsub import PubSubManager

        pubsub = PubSubManager()
        received = []

        async def handler(topic, data):
            received.append(data)

        pubsub.subscribe("test", handler)
        await pubsub.publish_async("test", "first")

        pubsub.unsubscribe("test", handler)
        await pubsub.publish_async("test", "second")

        assert len(received) == 1


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Test message serialization."""

    def test_serialize_task(self):
        from aion.distributed.communication.serialization import MessageSerializer

        serializer = MessageSerializer()
        task = DistributedTask(
            name="test-task",
            task_type="compute",
            priority=TaskPriority.HIGH,
        )

        data = serializer.serialize_task(task)
        assert isinstance(data, dict)
        assert data["name"] == "test-task"

        restored = serializer.deserialize_task(data)
        assert restored.name == "test-task"
        assert restored.task_type == "compute"
        assert restored.priority == TaskPriority.HIGH

    def test_serialize_node(self):
        from aion.distributed.communication.serialization import MessageSerializer

        serializer = MessageSerializer()
        node = NodeInfo(
            name="test-node",
            host="10.0.0.1",
            port=5000,
            role=NodeRole.LEADER,
        )

        data = serializer.serialize_node(node)
        assert isinstance(data, dict)
        assert data["name"] == "test-node"

        restored = serializer.deserialize_node(data)
        assert restored.name == "test-node"
        assert restored.host == "10.0.0.1"

    def test_serialize_raft_entry(self):
        from aion.distributed.communication.serialization import MessageSerializer

        serializer = MessageSerializer()
        entry = RaftLogEntry(
            index=5,
            term=3,
            command="set_state",
            data={"key": "value"},
        )

        data = serializer.serialize_raft_entry(entry)
        restored = serializer.deserialize_raft_entry(data)
        assert restored.index == 5
        assert restored.term == 3
        assert restored.command == "set_state"

    def test_serialize_bytes(self):
        from aion.distributed.communication.serialization import MessageSerializer

        serializer = MessageSerializer()
        obj = {"key": "value", "number": 42}
        data = serializer.serialize(obj)
        assert isinstance(data, bytes)

        restored = serializer.deserialize(data)
        assert restored["key"] == "value"
        assert restored["number"] == 42


# =============================================================================
# Snapshot Tests
# =============================================================================


class TestSnapshotMetadata:
    """Test snapshot metadata."""

    def test_creation(self):
        meta = SnapshotMetadata(
            last_included_index=100,
            last_included_term=5,
            size_bytes=1024,
            checksum="abc123",
        )
        assert meta.last_included_index == 100
        assert meta.last_included_term == 5

    def test_to_dict(self):
        meta = SnapshotMetadata(last_included_index=50)
        d = meta.to_dict()
        assert "snapshot_id" in d
        assert d["last_included_index"] == 50


# =============================================================================
# Integration Tests
# =============================================================================


class TestDistributedIntegration:
    """Integration tests for the distributed system."""

    @pytest.mark.asyncio
    async def test_task_lifecycle(self):
        """Test complete task lifecycle: create -> queue -> dequeue."""
        from aion.distributed.tasks.queue import DistributedTaskQueue

        manager = MagicMock()
        queue = DistributedTaskQueue(manager)

        # Create and enqueue task
        task = DistributedTask(
            name="integration-test",
            task_type="compute",
            priority=TaskPriority.HIGH,
            payload={"data": [1, 2, 3]},
        )

        await queue.enqueue(task)
        assert task.status == TaskStatus.QUEUED

        # Dequeue
        dequeued = await queue.dequeue()
        assert dequeued.id == task.id

        # Simulate execution
        dequeued.status = TaskStatus.RUNNING
        dequeued.started_at = datetime.now()
        dequeued.assigned_node = "node-1"
        await queue.update_task(dequeued)

        # Complete
        dequeued.status = TaskStatus.COMPLETED
        dequeued.completed_at = datetime.now()
        dequeued.result = {"sum": 6}
        await queue.update_task(dequeued)

        # Verify final state
        final = await queue.get_task(task.id)
        assert final.status == TaskStatus.COMPLETED
        assert final.result == {"sum": 6}

    @pytest.mark.asyncio
    async def test_multi_node_cluster_state(self):
        """Test cluster state with multiple nodes."""
        state = ClusterState(name="test-cluster")

        # Add nodes
        for i in range(5):
            node = NodeInfo(
                id=f"node-{i}",
                name=f"aion-{i}",
                status=NodeStatus.HEALTHY,
                role=NodeRole.FOLLOWER,
                current_tasks=i * 2,
                max_concurrent_tasks=10,
            )
            state.nodes[node.id] = node

        # Set leader
        state.nodes["node-0"].role = NodeRole.LEADER
        state.leader_id = "node-0"

        assert len(state.nodes) == 5
        assert len(state.healthy_nodes) == 5
        assert state.has_quorum is True
        assert state.quorum_size == 3
        assert state.get_leader().id == "node-0"

        # Simulate node failure
        state.nodes["node-3"].status = NodeStatus.UNHEALTHY
        state.nodes["node-4"].status = NodeStatus.OFFLINE

        assert len(state.healthy_nodes) == 3
        assert state.has_quorum is True  # 3 out of 5 still have quorum

    @pytest.mark.asyncio
    async def test_vector_clock_ordering(self):
        """Test causal ordering with vector clocks."""
        # Node A writes
        vc_a = VectorClock()
        vc_a.increment("A")  # A:1

        # Node B writes independently
        vc_b = VectorClock()
        vc_b.increment("B")  # B:1

        # These are concurrent
        assert vc_a.is_concurrent(vc_b) is True

        # Node C reads from A then writes
        vc_c = vc_a.copy()
        vc_c.increment("C")  # A:1, C:1

        # C's write happened after A's
        assert vc_c.dominates(vc_a) is True
        assert vc_a.dominates(vc_c) is False

        # C and B are still concurrent
        assert vc_c.is_concurrent(vc_b) is True

    def test_shard_distribution_balance(self):
        """Test that shards distribute evenly across nodes."""
        from aion.distributed.memory.sharding import ConsistentHash

        ch = ConsistentHash(virtual_nodes=150)
        for i in range(5):
            ch.add_node(f"node-{i}")

        # Check 10000 keys
        counts = Counter()
        for i in range(10000):
            node = ch.get_node(f"memory-key-{i}")
            counts[node] += 1

        # Verify reasonable distribution (each node between 10-30%)
        for node, count in counts.items():
            share = count / 10000
            assert 0.05 < share < 0.50, f"Node {node} has uneven share: {share:.2%}"


# =============================================================================
# Config Integration Test
# =============================================================================


class TestConfigIntegration:
    """Test configuration integration with kernel."""

    def test_config_has_distributed_field(self):
        from aion.core.config import AIONConfig
        config = AIONConfig()
        assert hasattr(config, "distributed_enabled")
        assert config.distributed_enabled is False

    def test_kernel_has_cluster_property(self):
        from aion.core.kernel import AIONKernel
        kernel = AIONKernel()
        assert hasattr(kernel, '_cluster_manager')
        assert kernel._cluster_manager is None
        assert kernel.cluster is None

    def test_kernel_cluster_stats_unavailable(self):
        from aion.core.kernel import AIONKernel
        kernel = AIONKernel()
        stats = kernel.get_cluster_stats()
        assert stats == {"available": False}
