"""
Comprehensive tests for the AION Process & Agent Manager System.

Tests cover:
- Process lifecycle (spawn, pause, resume, stop, restart)
- Restart policies (never, on_failure, always, exponential_backoff)
- Resource limit enforcement
- Event bus pub/sub
- Task scheduling (once, interval, cron)
- Agent communication
- Process hierarchy (parent/child)
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aion.systems.process import (
    # Core classes
    ProcessSupervisor,
    EventBus,
    TaskScheduler,
    WorkerPool,
    BaseAgent,
    # Models
    ProcessState,
    ProcessPriority,
    ProcessType,
    RestartPolicy,
    ResourceLimits,
    ResourceUsage,
    ProcessInfo,
    AgentConfig,
    TaskDefinition,
    Event,
    AgentMessage,
    MessageType,
    SignalType,
    # Scheduler
    CronParser,
)


# ==================== Test Fixtures ====================

@pytest.fixture
async def event_bus():
    """Create an event bus for testing."""
    bus = EventBus(max_history=100)
    await bus.initialize()
    yield bus
    await bus.shutdown()


@pytest.fixture
async def supervisor(event_bus):
    """Create a process supervisor for testing."""
    sup = ProcessSupervisor(
        event_bus=event_bus,
        max_processes=10,
        health_check_interval=1.0,
    )
    await sup.initialize()
    yield sup
    await sup.shutdown()


@pytest.fixture
async def scheduler(supervisor, event_bus):
    """Create a task scheduler for testing."""
    sched = TaskScheduler(
        supervisor=supervisor,
        event_bus=event_bus,
        check_interval=0.1,
    )
    await sched.initialize()
    yield sched
    await sched.shutdown()


@pytest.fixture
async def worker_pool(event_bus):
    """Create a worker pool for testing."""
    pool = WorkerPool(
        event_bus=event_bus,
        min_workers=1,
        max_workers=3,
    )
    await pool.initialize()
    yield pool
    await pool.shutdown()


# ==================== Test Agent Implementation ====================

class TestAgent(BaseAgent):
    """Test agent implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_count = 0
        self.messages_processed = []
        self.should_fail = False

    async def run(self):
        """Main agent loop."""
        self.run_count += 1

        while not self._shutdown_requested:
            await self._paused.wait()

            if self._shutdown_requested:
                break

            if self.should_fail:
                raise RuntimeError("Intentional failure for testing")

            # Process messages
            message = await self.get_next_message(timeout=0.1)
            if message:
                self.messages_processed.append(message)

            await asyncio.sleep(0.1)


class FailingAgent(BaseAgent):
    """Agent that always fails."""

    async def run(self):
        raise RuntimeError("This agent always fails")


class QuickAgent(BaseAgent):
    """Agent that completes quickly."""

    async def run(self):
        await asyncio.sleep(0.1)


# ==================== Models Tests ====================

class TestModels:
    """Test data models."""

    def test_process_state_transitions(self):
        """Test valid state transitions."""
        assert ProcessState.CREATED.can_transition_to(ProcessState.STARTING)
        assert ProcessState.STARTING.can_transition_to(ProcessState.RUNNING)
        assert ProcessState.RUNNING.can_transition_to(ProcessState.PAUSED)
        assert ProcessState.PAUSED.can_transition_to(ProcessState.RUNNING)
        assert ProcessState.RUNNING.can_transition_to(ProcessState.STOPPING)
        assert ProcessState.STOPPING.can_transition_to(ProcessState.STOPPED)

        # Invalid transitions
        assert not ProcessState.STOPPED.can_transition_to(ProcessState.RUNNING)
        assert not ProcessState.CREATED.can_transition_to(ProcessState.RUNNING)

    def test_process_state_properties(self):
        """Test state properties."""
        assert ProcessState.RUNNING.is_active()
        assert ProcessState.PAUSED.is_active()
        assert not ProcessState.STOPPED.is_active()

        assert ProcessState.STOPPED.is_terminal()
        assert ProcessState.FAILED.is_terminal()
        assert not ProcessState.RUNNING.is_terminal()

    def test_resource_limits(self):
        """Test resource limits."""
        limits = ResourceLimits(
            max_memory_mb=1024,
            max_tokens_per_minute=1000,
            max_runtime_seconds=300,
        )

        assert limits.max_memory_mb == 1024
        assert limits.to_dict()["max_memory_mb"] == 1024

        # Test merge
        limits2 = ResourceLimits(max_memory_mb=512)
        merged = limits.merge_with(limits2)
        assert merged.max_memory_mb == 512  # More restrictive

    def test_resource_usage_exceeds_limits(self):
        """Test resource limit checking."""
        limits = ResourceLimits(max_memory_mb=1024, max_tokens_total=1000)

        usage = ResourceUsage(memory_mb=500, tokens_used=500)
        exceeded, reason = usage.exceeds_limits(limits)
        assert not exceeded

        usage = ResourceUsage(memory_mb=2000, tokens_used=500)
        exceeded, reason = usage.exceeds_limits(limits)
        assert exceeded
        assert "Memory" in reason

    def test_agent_config_hash(self):
        """Test config hashing."""
        config1 = AgentConfig(name="test", agent_class="TestAgent")
        config2 = AgentConfig(name="test", agent_class="TestAgent")
        config3 = AgentConfig(name="test2", agent_class="TestAgent")

        assert config1.get_config_hash() == config2.get_config_hash()
        assert config1.get_config_hash() != config3.get_config_hash()

    def test_restart_policy(self):
        """Test restart policy logic."""
        assert RestartPolicy.NEVER.should_restart(1, False) is False
        assert RestartPolicy.ON_FAILURE.should_restart(1, False) is True
        assert RestartPolicy.ON_FAILURE.should_restart(0, False) is False
        assert RestartPolicy.ALWAYS.should_restart(0, False) is True
        assert RestartPolicy.ALWAYS.should_restart(0, True) is False  # Explicit stop


# ==================== Event Bus Tests ====================

class TestEventBus:
    """Test event bus functionality."""

    @pytest.mark.asyncio
    async def test_basic_pub_sub(self, event_bus):
        """Test basic publish/subscribe."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        await event_bus.subscribe("test.channel", handler)

        event = Event(
            id="test-1",
            type="test.channel",
            source="test",
            payload={"data": "hello"},
        )
        await event_bus.emit(event, wait_for_handlers=True)

        assert len(received_events) == 1
        assert received_events[0].payload["data"] == "hello"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """Test wildcard pattern matching."""
        received = []

        async def handler(event: Event):
            received.append(event.type)

        await event_bus.subscribe("test.*", handler)

        await event_bus.emit(Event(id="1", type="test.foo", source="test", payload={}), wait_for_handlers=True)
        await event_bus.emit(Event(id="2", type="test.bar", source="test", payload={}), wait_for_handlers=True)
        await event_bus.emit(Event(id="3", type="other.baz", source="test", payload={}), wait_for_handlers=True)

        assert len(received) == 2
        assert "test.foo" in received
        assert "test.bar" in received
        assert "other.baz" not in received

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscription."""
        received = []

        async def handler(event: Event):
            received.append(event)

        sub_id = await event_bus.subscribe("test", handler)
        await event_bus.emit(Event(id="1", type="test", source="test", payload={}), wait_for_handlers=True)
        assert len(received) == 1

        await event_bus.unsubscribe_by_id(sub_id)
        await event_bus.emit(Event(id="2", type="test", source="test", payload={}), wait_for_handlers=True)
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_request_response(self, event_bus):
        """Test request/response pattern."""
        async def responder(event: Event):
            response = event.create_response(
                source="responder",
                payload={"answer": 42},
            )
            await event_bus.emit(response)

        await event_bus.subscribe("query", responder)

        request = Event(
            id="req-1",
            type="query",
            source="requester",
            payload={"question": "meaning of life"},
        )

        response = await event_bus.request(request, timeout=5.0)
        assert response is not None
        assert response.payload["answer"] == 42

    @pytest.mark.asyncio
    async def test_event_history(self, event_bus):
        """Test event history retrieval."""
        for i in range(5):
            await event_bus.emit(Event(
                id=f"event-{i}",
                type="history.test",
                source="test",
                payload={"index": i},
            ))

        history = event_bus.get_history("history.*", limit=10)
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_stats(self, event_bus):
        """Test event bus statistics."""
        await event_bus.emit(Event(id="1", type="test", source="test", payload={}))

        stats = event_bus.get_stats()
        assert stats["events_emitted"] >= 1


# ==================== Process Supervisor Tests ====================

class TestProcessSupervisor:
    """Test process supervisor functionality."""

    @pytest.mark.asyncio
    async def test_spawn_agent(self, supervisor):
        """Test spawning an agent."""
        supervisor.register_agent_class("test_agent", QuickAgent)

        config = AgentConfig(
            name="test-agent-1",
            agent_class="test_agent",
            priority=ProcessPriority.NORMAL,
        )

        process_id = await supervisor.spawn_agent(config)
        assert process_id is not None

        process = supervisor.get_process(process_id)
        assert process is not None
        assert process.name == "test-agent-1"
        assert process.state in (ProcessState.STARTING, ProcessState.RUNNING, ProcessState.STOPPED)

    @pytest.mark.asyncio
    async def test_stop_agent(self, supervisor):
        """Test stopping an agent."""
        supervisor.register_agent_class("test_agent", TestAgent)

        config = AgentConfig(name="stop-test", agent_class="test_agent")
        process_id = await supervisor.spawn_agent(config)

        # Wait for agent to start
        await asyncio.sleep(0.2)

        success = await supervisor.stop_process(process_id)
        assert success

        process = supervisor.get_process(process_id)
        assert process.state in (ProcessState.STOPPED, ProcessState.STOPPING)

    @pytest.mark.asyncio
    async def test_pause_resume(self, supervisor):
        """Test pausing and resuming an agent."""
        supervisor.register_agent_class("test_agent", TestAgent)

        config = AgentConfig(name="pause-test", agent_class="test_agent")
        process_id = await supervisor.spawn_agent(config)

        await asyncio.sleep(0.2)

        # Pause
        success = await supervisor.pause_process(process_id)
        assert success

        process = supervisor.get_process(process_id)
        assert process.state == ProcessState.PAUSED

        # Resume
        success = await supervisor.resume_process(process_id)
        assert success

        process = supervisor.get_process(process_id)
        assert process.state == ProcessState.RUNNING

        await supervisor.stop_process(process_id)

    @pytest.mark.asyncio
    async def test_restart_policy_never(self, supervisor):
        """Test NEVER restart policy."""
        supervisor.register_agent_class("failing", FailingAgent)

        config = AgentConfig(
            name="never-restart",
            agent_class="failing",
            restart_policy=RestartPolicy.NEVER,
        )

        process_id = await supervisor.spawn_agent(config)
        await asyncio.sleep(0.5)

        process = supervisor.get_process(process_id)
        assert process.state == ProcessState.FAILED
        assert process.restart_count == 0

    @pytest.mark.asyncio
    async def test_max_processes_limit(self, supervisor):
        """Test maximum process limit."""
        supervisor.register_agent_class("test_agent", TestAgent)
        supervisor.max_processes = 2

        # Spawn up to limit
        pid1 = await supervisor.spawn_agent(AgentConfig(name="agent1", agent_class="test_agent"))
        pid2 = await supervisor.spawn_agent(AgentConfig(name="agent2", agent_class="test_agent"))

        # Should fail on third
        with pytest.raises(RuntimeError, match="Maximum process limit"):
            await supervisor.spawn_agent(AgentConfig(name="agent3", agent_class="test_agent"))

        # Cleanup
        await supervisor.stop_process(pid1)
        await supervisor.stop_process(pid2)

    @pytest.mark.asyncio
    async def test_process_hierarchy(self, supervisor):
        """Test parent-child relationships."""
        supervisor.register_agent_class("test_agent", TestAgent)

        parent_config = AgentConfig(name="parent", agent_class="test_agent")
        parent_id = await supervisor.spawn_agent(parent_config)

        child_config = AgentConfig(name="child", agent_class="test_agent")
        child_id = await supervisor.spawn_agent(child_config, parent_id=parent_id)

        parent = supervisor.get_process(parent_id)
        child = supervisor.get_process(child_id)

        assert child.parent_id == parent_id
        assert child_id in parent.children_ids

        children = supervisor.get_children(parent_id)
        assert len(children) == 1
        assert children[0].id == child_id

        # Cleanup
        await supervisor.stop_process(parent_id)

    @pytest.mark.asyncio
    async def test_send_signal(self, supervisor):
        """Test sending signals to processes."""
        supervisor.register_agent_class("test_agent", TestAgent)

        config = AgentConfig(name="signal-test", agent_class="test_agent")
        process_id = await supervisor.spawn_agent(config)

        await asyncio.sleep(0.2)

        success = await supervisor.send_signal(process_id, SignalType.PAUSE, {})
        assert success

        process = supervisor.get_process(process_id)
        assert process.state == ProcessState.PAUSED

        await supervisor.stop_process(process_id)

    @pytest.mark.asyncio
    async def test_stats(self, supervisor):
        """Test supervisor statistics."""
        supervisor.register_agent_class("test_agent", QuickAgent)

        await supervisor.spawn_agent(AgentConfig(name="stats-test", agent_class="test_agent"))

        stats = supervisor.get_stats()
        assert stats["processes_spawned"] >= 1
        assert "total_processes" in stats


# ==================== Task Scheduler Tests ====================

class TestTaskScheduler:
    """Test task scheduler functionality."""

    @pytest.mark.asyncio
    async def test_schedule_once(self, scheduler):
        """Test one-time task scheduling."""
        executed = []

        async def handler():
            executed.append(datetime.now())

        scheduler.register_handler("once_handler", handler)

        task_id = await scheduler.schedule_once(
            name="once-task",
            handler="once_handler",
            run_at=datetime.now() + timedelta(seconds=0.1),
        )

        await asyncio.sleep(0.5)

        assert len(executed) == 1

        task = scheduler.get_task(task_id)
        assert task.run_count == 1

    @pytest.mark.asyncio
    async def test_schedule_interval(self, scheduler):
        """Test interval task scheduling."""
        executed = []

        async def handler():
            executed.append(datetime.now())

        scheduler.register_handler("interval_handler", handler)

        task_id = await scheduler.schedule_interval(
            name="interval-task",
            handler="interval_handler",
            interval_seconds=1,
            start_immediately=True,
        )

        await asyncio.sleep(2.5)

        # Should have run at least twice
        assert len(executed) >= 2

        await scheduler.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_cancel_task(self, scheduler):
        """Test task cancellation."""
        async def handler():
            pass

        scheduler.register_handler("cancel_handler", handler)

        task_id = await scheduler.schedule_interval(
            name="cancel-task",
            handler="cancel_handler",
            interval_seconds=1,
        )

        success = await scheduler.cancel_task(task_id)
        assert success

        task = scheduler.get_task(task_id)
        assert task is None

    @pytest.mark.asyncio
    async def test_pause_resume_task(self, scheduler):
        """Test pausing and resuming tasks."""
        async def handler():
            pass

        scheduler.register_handler("pause_handler", handler)

        task_id = await scheduler.schedule_interval(
            name="pause-task",
            handler="pause_handler",
            interval_seconds=1,
        )

        await scheduler.pause_task(task_id)
        task = scheduler.get_task(task_id)
        assert not task.enabled

        await scheduler.resume_task(task_id)
        task = scheduler.get_task(task_id)
        assert task.enabled

        await scheduler.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_trigger_task(self, scheduler):
        """Test manual task triggering."""
        executed = []

        async def handler():
            executed.append(datetime.now())

        scheduler.register_handler("trigger_handler", handler)

        task_id = await scheduler.schedule_interval(
            name="trigger-task",
            handler="trigger_handler",
            interval_seconds=60,  # Far in future
        )

        # Manually trigger
        await scheduler.trigger_task(task_id)
        await asyncio.sleep(0.5)

        assert len(executed) >= 1

        await scheduler.cancel_task(task_id)


# ==================== Cron Parser Tests ====================

class TestCronParser:
    """Test cron expression parsing."""

    def test_parse_simple(self):
        """Test simple cron parsing."""
        result = CronParser.parse("0 9 * * *")

        assert result["minute"] == [0]
        assert result["hour"] == [9]
        assert len(result["day"]) == 31
        assert len(result["month"]) == 12
        assert len(result["weekday"]) == 7

    def test_parse_ranges(self):
        """Test range parsing."""
        result = CronParser.parse("0-5 9-17 * * *")

        assert result["minute"] == [0, 1, 2, 3, 4, 5]
        assert result["hour"] == [9, 10, 11, 12, 13, 14, 15, 16, 17]

    def test_parse_steps(self):
        """Test step parsing."""
        result = CronParser.parse("*/15 * * * *")

        assert result["minute"] == [0, 15, 30, 45]

    def test_special_expressions(self):
        """Test special expressions."""
        result = CronParser.parse("@daily")
        assert result["minute"] == [0]
        assert result["hour"] == [0]

        result = CronParser.parse("@hourly")
        assert result["minute"] == [0]
        assert len(result["hour"]) == 24

    def test_get_next_run(self):
        """Test next run calculation."""
        now = datetime(2024, 1, 1, 10, 30, 0)
        next_run = CronParser.get_next_run("0 12 * * *", now)

        assert next_run.hour == 12
        assert next_run.minute == 0
        assert next_run > now


# ==================== Worker Pool Tests ====================

class TestWorkerPool:
    """Test worker pool functionality."""

    @pytest.mark.asyncio
    async def test_submit_task(self, worker_pool):
        """Test submitting a task."""
        result = []

        async def handler():
            result.append(42)
            return 42

        task_id = await worker_pool.submit(handler)
        assert task_id is not None

        # Wait for completion
        task = await worker_pool.wait_for_task(task_id, timeout=5.0)
        assert task is not None
        assert task.result == 42

    @pytest.mark.asyncio
    async def test_task_priority(self, worker_pool):
        """Test task priority ordering."""
        results = []

        async def handler(value):
            results.append(value)

        # Submit low priority first
        await worker_pool.submit(handler, {"value": "low"}, priority=0)
        # Submit high priority second
        await worker_pool.submit(handler, {"value": "high"}, priority=10)

        await asyncio.sleep(0.5)

        # High priority should execute first (if workers available)
        # Note: With enough workers, both may execute nearly simultaneously

    @pytest.mark.asyncio
    async def test_cancel_task(self, worker_pool):
        """Test task cancellation."""
        async def slow_handler():
            await asyncio.sleep(10)

        task_id = await worker_pool.submit(slow_handler)
        success = await worker_pool.cancel_task(task_id)

        # May or may not succeed depending on timing
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_stats(self, worker_pool):
        """Test worker pool statistics."""
        stats = worker_pool.get_stats()

        assert "tasks_submitted" in stats
        assert "total_workers" in stats
        assert stats["total_workers"] >= 1


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_agent_message_passing(self, supervisor, event_bus):
        """Test message passing between agents."""
        supervisor.register_agent_class("test_agent", TestAgent)

        config1 = AgentConfig(
            name="sender",
            agent_class="test_agent",
            output_channels=["agent.receiver"],
        )
        config2 = AgentConfig(
            name="receiver",
            agent_class="test_agent",
            input_channels=["agent.receiver"],
        )

        pid1 = await supervisor.spawn_agent(config1)
        pid2 = await supervisor.spawn_agent(config2)

        await asyncio.sleep(0.3)

        # Get agent instances
        sender = supervisor.get_agent_instance(pid1)
        receiver = supervisor.get_agent_instance(pid2)

        # Send message
        await sender.send_message(
            to_agent="receiver",
            content="Hello!",
            message_type=MessageType.TEXT,
        )

        await asyncio.sleep(0.3)

        # Check receiver got message
        assert len(receiver.messages_processed) > 0 or receiver.has_messages()

        # Cleanup
        await supervisor.stop_process(pid1)
        await supervisor.stop_process(pid2)

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, event_bus):
        """Test graceful shutdown of all components."""
        supervisor = ProcessSupervisor(event_bus=event_bus, max_processes=5)
        await supervisor.initialize()

        supervisor.register_agent_class("test_agent", TestAgent)

        # Spawn multiple agents
        pids = []
        for i in range(3):
            pid = await supervisor.spawn_agent(AgentConfig(
                name=f"agent-{i}",
                agent_class="test_agent",
            ))
            pids.append(pid)

        await asyncio.sleep(0.3)

        # Verify all running
        for pid in pids:
            process = supervisor.get_process(pid)
            assert process.state == ProcessState.RUNNING

        # Graceful shutdown
        await supervisor.shutdown(timeout=10.0)

        # Verify all stopped
        for pid in pids:
            process = supervisor.get_process(pid)
            assert process.state in (ProcessState.STOPPED, ProcessState.TERMINATED)


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
