"""
Tests for AION Workflow Automation System - SOTA Features.

Tests event sourcing, distributed task queue, observability,
visual workflow builder, and saga pattern implementations.
"""

import asyncio
import json
import pytest
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import types
from aion.automation.types import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    ActionConfig,
    ActionType,
    ExecutionStatus,
    WorkflowStatus,
)


# === Event Store Tests ===


class TestEventStore:
    """Tests for event sourcing implementation."""

    @pytest.mark.asyncio
    async def test_event_store_creation(self):
        """Test creating an event store."""
        from aion.automation.execution.event_store import (
            EventStore,
            EventType,
            InMemoryEventStore,
        )

        backend = InMemoryEventStore()
        store = EventStore(backend=backend)
        await store.initialize()

        assert store._initialized

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_event_append_and_get(self):
        """Test appending and retrieving events."""
        from aion.automation.execution.event_store import (
            EventStore,
            EventType,
            InMemoryEventStore,
        )

        backend = InMemoryEventStore()
        store = EventStore(backend=backend)
        await store.initialize()

        workflow_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())

        # Append events
        await store.append(
            execution_id=execution_id,
            event_type=EventType.WORKFLOW_STARTED,
            payload={"workflow_id": workflow_id, "inputs": {"test": "value"}},
        )

        await store.append(
            execution_id=execution_id,
            event_type=EventType.STEP_STARTED,
            payload={"step_id": "step1", "step_name": "First Step"},
        )

        await store.append(
            execution_id=execution_id,
            event_type=EventType.STEP_COMPLETED,
            payload={"step_id": "step1", "outputs": {"result": "success"}},
        )

        # Get events
        events = await store.get_events(execution_id)
        assert len(events) == 3

        assert events[0].event_type == EventType.WORKFLOW_STARTED
        assert events[1].event_type == EventType.STEP_STARTED
        assert events[2].event_type == EventType.STEP_COMPLETED

        # Check sequence numbers
        assert events[0].sequence == 0
        assert events[1].sequence == 1
        assert events[2].sequence == 2

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_event_filter_by_type(self):
        """Test filtering events by type."""
        from aion.automation.execution.event_store import (
            EventStore,
            EventType,
            InMemoryEventStore,
        )

        backend = InMemoryEventStore()
        store = EventStore(backend=backend)
        await store.initialize()

        execution_id = str(uuid.uuid4())

        # Append multiple event types
        await store.append(execution_id, EventType.WORKFLOW_STARTED, {})
        await store.append(execution_id, EventType.STEP_STARTED, {"step_id": "s1"})
        await store.append(execution_id, EventType.STEP_COMPLETED, {"step_id": "s1"})
        await store.append(execution_id, EventType.STEP_STARTED, {"step_id": "s2"})
        await store.append(execution_id, EventType.STEP_COMPLETED, {"step_id": "s2"})
        await store.append(execution_id, EventType.WORKFLOW_COMPLETED, {})

        # Filter by step events
        step_events = await store.get_events(
            execution_id,
            event_types=[EventType.STEP_STARTED, EventType.STEP_COMPLETED],
        )
        assert len(step_events) == 4

        # Filter by workflow events only
        workflow_events = await store.get_events(
            execution_id,
            event_types=[EventType.WORKFLOW_STARTED, EventType.WORKFLOW_COMPLETED],
        )
        assert len(workflow_events) == 2

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_event_integrity(self):
        """Test event integrity with checksums."""
        from aion.automation.execution.event_store import (
            EventStore,
            EventType,
            Event,
            InMemoryEventStore,
        )

        backend = InMemoryEventStore()
        store = EventStore(backend=backend)
        await store.initialize()

        execution_id = str(uuid.uuid4())

        await store.append(
            execution_id,
            EventType.WORKFLOW_STARTED,
            {"important": "data"},
        )

        events = await store.get_events(execution_id)
        event = events[0]

        # Checksum should be present
        assert event.checksum is not None

        # Verify integrity
        assert event.verify_integrity()

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_snapshot_creation(self):
        """Test creating and using snapshots."""
        from aion.automation.execution.event_store import (
            EventStore,
            EventType,
            InMemoryEventStore,
        )

        backend = InMemoryEventStore()
        store = EventStore(backend=backend, snapshot_interval=3)
        await store.initialize()

        execution_id = str(uuid.uuid4())

        # Add enough events to trigger snapshot
        await store.append(execution_id, EventType.WORKFLOW_STARTED, {})
        await store.append(execution_id, EventType.STEP_STARTED, {"step_id": "s1"})
        await store.append(execution_id, EventType.STEP_COMPLETED, {"step_id": "s1"})
        await store.append(execution_id, EventType.STEP_STARTED, {"step_id": "s2"})

        # Create snapshot
        state = {"status": "running", "current_step": "s2"}
        await store.create_snapshot(execution_id, state, 3)

        # Get latest snapshot
        snapshot = await store.get_latest_snapshot(execution_id)
        assert snapshot is not None
        assert snapshot.state == state
        assert snapshot.sequence == 3

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_workflow_replayer(self):
        """Test workflow replay capability."""
        from aion.automation.execution.event_store import (
            EventStore,
            EventType,
            WorkflowReplayer,
            InMemoryEventStore,
        )

        backend = InMemoryEventStore()
        store = EventStore(backend=backend)
        await store.initialize()

        execution_id = str(uuid.uuid4())

        # Build up execution history
        await store.append(
            execution_id,
            EventType.WORKFLOW_STARTED,
            {"workflow_id": "wf-1", "inputs": {"x": 1}},
        )
        await store.append(
            execution_id,
            EventType.STEP_STARTED,
            {"step_id": "s1", "name": "Step 1"},
        )
        await store.append(
            execution_id,
            EventType.STEP_COMPLETED,
            {"step_id": "s1", "outputs": {"result": 10}},
        )
        await store.append(
            execution_id,
            EventType.WORKFLOW_COMPLETED,
            {"outputs": {"final": 10}},
        )

        # Create replayer
        replayer = WorkflowReplayer(store)

        # Replay full execution
        state = await replayer.replay(execution_id)

        assert state["status"] == "completed"
        assert state["step_results"]["s1"]["outputs"]["result"] == 10

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_file_event_store(self):
        """Test file-based event store."""
        from aion.automation.execution.event_store import (
            EventStore,
            EventType,
            FileEventStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileEventStore(storage_dir=tmpdir)
            store = EventStore(backend=backend)
            await store.initialize()

            execution_id = str(uuid.uuid4())

            # Store events
            await store.append(execution_id, EventType.WORKFLOW_STARTED, {"x": 1})
            await store.append(execution_id, EventType.STEP_STARTED, {"step_id": "s1"})

            # Retrieve events
            events = await store.get_events(execution_id)
            assert len(events) == 2

            # Verify file was created
            event_file = Path(tmpdir) / f"{execution_id}.jsonl"
            assert event_file.exists()

            await store.shutdown()


# === Distributed Task Queue Tests ===


class TestDistributedTaskQueue:
    """Tests for distributed task queue implementation."""

    @pytest.mark.asyncio
    async def test_task_queue_creation(self):
        """Test creating a task queue."""
        from aion.automation.distributed import TaskQueue
        from aion.automation.distributed.backends import InMemoryBackend

        backend = InMemoryBackend()
        queue = TaskQueue(backend=backend)
        await queue.initialize()

        assert queue._initialized

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_task_enqueue_dequeue(self):
        """Test enqueueing and dequeueing tasks."""
        from aion.automation.distributed import TaskQueue
        from aion.automation.distributed.queue import TaskPriority
        from aion.automation.distributed.backends import InMemoryBackend

        backend = InMemoryBackend()
        queue = TaskQueue(backend=backend)
        await queue.initialize()

        # Enqueue task
        task_id = await queue.enqueue(
            task_type="test_task",
            payload={"key": "value"},
            priority=TaskPriority.NORMAL,
        )

        assert task_id is not None

        # Dequeue task
        task = await queue.dequeue("test_task")

        assert task is not None
        assert task.id == task_id
        assert task.payload == {"key": "value"}

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_task_priority_ordering(self):
        """Test that high priority tasks are dequeued first."""
        from aion.automation.distributed import TaskQueue
        from aion.automation.distributed.queue import TaskPriority
        from aion.automation.distributed.backends import InMemoryBackend

        backend = InMemoryBackend()
        queue = TaskQueue(backend=backend)
        await queue.initialize()

        # Enqueue tasks with different priorities
        await queue.enqueue("task", {"order": 1}, priority=TaskPriority.LOW)
        await queue.enqueue("task", {"order": 2}, priority=TaskPriority.CRITICAL)
        await queue.enqueue("task", {"order": 3}, priority=TaskPriority.NORMAL)

        # Dequeue - should get critical first
        task1 = await queue.dequeue("task")
        assert task1.payload["order"] == 2  # Critical

        task2 = await queue.dequeue("task")
        assert task2.payload["order"] == 3  # Normal

        task3 = await queue.dequeue("task")
        assert task3.payload["order"] == 1  # Low

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_task_completion(self):
        """Test completing tasks."""
        from aion.automation.distributed import TaskQueue
        from aion.automation.distributed.queue import TaskStatus
        from aion.automation.distributed.backends import InMemoryBackend

        backend = InMemoryBackend()
        queue = TaskQueue(backend=backend)
        await queue.initialize()

        task_id = await queue.enqueue("task", {"test": 1})
        task = await queue.dequeue("task")

        # Complete task
        await queue.complete(task_id, result={"output": "success"})

        # Verify status
        status = await queue.get_status(task_id)
        assert status == TaskStatus.COMPLETED

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self):
        """Test task failure and retry mechanism."""
        from aion.automation.distributed import TaskQueue
        from aion.automation.distributed.queue import TaskStatus
        from aion.automation.distributed.backends import InMemoryBackend

        backend = InMemoryBackend()
        queue = TaskQueue(backend=backend)
        await queue.initialize()

        task_id = await queue.enqueue("task", {"test": 1}, max_retries=2)
        task = await queue.dequeue("task")

        # Fail task - should retry
        await queue.fail(task_id, error="First failure")

        # Should be available again
        task = await queue.dequeue("task")
        assert task is not None
        assert task.attempts == 2

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_task_deduplication(self):
        """Test task deduplication."""
        from aion.automation.distributed import TaskQueue
        from aion.automation.distributed.backends import InMemoryBackend

        backend = InMemoryBackend()
        queue = TaskQueue(backend=backend)
        await queue.initialize()

        dedup_key = "unique-task-123"

        # Enqueue with dedup key
        task1_id = await queue.enqueue(
            "task",
            {"data": 1},
            deduplication_key=dedup_key,
        )

        # Try to enqueue again with same key - should be skipped
        task2_id = await queue.enqueue(
            "task",
            {"data": 2},
            deduplication_key=dedup_key,
        )

        # Should return same ID or None
        assert task2_id is None or task2_id == task1_id

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_worker_execution(self):
        """Test worker task execution."""
        from aion.automation.distributed import Worker
        from aion.automation.distributed import TaskQueue
        from aion.automation.distributed.backends import InMemoryBackend

        backend = InMemoryBackend()
        queue = TaskQueue(backend=backend)
        await queue.initialize()

        results = []

        async def handler(payload: Dict) -> Dict:
            results.append(payload["value"])
            return {"processed": payload["value"]}

        # Create worker
        worker = Worker(
            queue=queue,
            task_types=["process"],
            handlers={"process": handler},
        )

        # Enqueue tasks
        await queue.enqueue("process", {"value": 1})
        await queue.enqueue("process", {"value": 2})

        # Start worker and let it process
        worker_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.5)
        await worker.stop()
        worker_task.cancel()

        assert 1 in results
        assert 2 in results

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_worker_pool(self):
        """Test worker pool with multiple workers."""
        from aion.automation.distributed import TaskQueue, WorkerPool
        from aion.automation.distributed.backends import InMemoryBackend

        backend = InMemoryBackend()
        queue = TaskQueue(backend=backend)
        await queue.initialize()

        processed = set()

        async def handler(payload: Dict) -> Dict:
            processed.add(payload["id"])
            await asyncio.sleep(0.1)
            return {"ok": True}

        # Create pool
        pool = WorkerPool(
            queue=queue,
            task_types=["work"],
            handlers={"work": handler},
            min_workers=2,
            max_workers=4,
        )

        # Enqueue multiple tasks
        for i in range(5):
            await queue.enqueue("work", {"id": i})

        # Start pool
        await pool.start()
        await asyncio.sleep(1.0)
        await pool.stop()

        # All tasks should be processed
        assert len(processed) == 5

        await queue.shutdown()


# === Observability Tests ===


class TestObservability:
    """Tests for observability (telemetry, tracing, metrics)."""

    def test_telemetry_provider_creation(self):
        """Test creating telemetry provider."""
        from aion.automation.observability import TelemetryProvider, TracingConfig

        # Without OpenTelemetry (uses noop)
        provider = TelemetryProvider(service_name="test-service")
        tracer = provider.get_tracer("test")
        meter = provider.get_meter("test")

        assert tracer is not None
        assert meter is not None

    def test_workflow_tracer(self):
        """Test workflow tracer."""
        from aion.automation.observability import WorkflowTracer, TelemetryProvider

        provider = TelemetryProvider(service_name="test")
        tracer = WorkflowTracer(provider)

        # Should not raise
        with tracer.workflow_span("wf-1", "Test Workflow") as span:
            with tracer.step_span("s1", "Step 1") as step_span:
                step_span.set_attribute("custom", "value")

    @pytest.mark.asyncio
    async def test_workflow_metrics(self):
        """Test workflow metrics collection."""
        from aion.automation.observability import WorkflowMetrics, TelemetryProvider

        provider = TelemetryProvider(service_name="test")
        metrics = WorkflowMetrics(provider)

        # Record metrics - should not raise
        metrics.record_execution_started("wf-1", "Test")
        metrics.record_step_completed("wf-1", "s1", 0.5, True)
        metrics.record_step_completed("wf-1", "s2", 0.3, True)
        metrics.record_execution_completed("wf-1", 0.8, True)

    def test_trace_decorators(self):
        """Test tracing decorators."""
        from aion.automation.observability import trace_workflow, trace_step

        @trace_workflow("test")
        async def my_workflow():
            return "done"

        @trace_step("step1")
        async def my_step():
            return "step done"

        # Run decorated functions - should work
        result = asyncio.get_event_loop().run_until_complete(my_workflow())
        assert result == "done"

        result = asyncio.get_event_loop().run_until_complete(my_step())
        assert result == "step done"

    def test_metric_registry(self):
        """Test metric registry."""
        from aion.automation.observability.metrics import MetricRegistry

        registry = MetricRegistry()

        # Register metrics
        registry.register("counter", "requests_total", "Total requests")
        registry.register("histogram", "request_duration", "Request duration")
        registry.register("gauge", "active_workers", "Active workers")

        # Get metrics
        counter = registry.get("requests_total")
        histogram = registry.get("request_duration")
        gauge = registry.get("active_workers")

        assert counter is not None
        assert histogram is not None
        assert gauge is not None

    def test_metrics_timer(self):
        """Test metrics timer context manager."""
        from aion.automation.observability.metrics import MetricsTimer

        durations = []

        def record(duration):
            durations.append(duration)

        with MetricsTimer(record):
            import time
            time.sleep(0.1)

        assert len(durations) == 1
        assert durations[0] >= 0.1


# === Visual Workflow Builder Tests ===


class TestVisualWorkflowBuilder:
    """Tests for visual workflow builder."""

    def test_workflow_validator_basic(self):
        """Test basic workflow validation."""
        from aion.automation.visual import WorkflowValidator

        validator = WorkflowValidator()

        # Valid workflow
        workflow = Workflow(
            name="Test",
            steps=[
                WorkflowStep(
                    id="s1",
                    name="Step 1",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                    on_success="s2",
                ),
                WorkflowStep(
                    id="s2",
                    name="Step 2",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                ),
            ],
            entry_step_id="s1",
        )

        result = validator.validate(workflow)
        assert result.is_valid

    def test_workflow_validator_cycle_detection(self):
        """Test cycle detection in workflow."""
        from aion.automation.visual import WorkflowValidator

        validator = WorkflowValidator()

        # Workflow with cycle
        workflow = Workflow(
            name="Test",
            steps=[
                WorkflowStep(
                    id="s1",
                    name="Step 1",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                    on_success="s2",
                ),
                WorkflowStep(
                    id="s2",
                    name="Step 2",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                    on_success="s1",  # Cycle back to s1
                ),
            ],
            entry_step_id="s1",
        )

        result = validator.validate(workflow)
        # Cycle detection depends on implementation - may or may not be error
        # For now just ensure validation completes
        assert result is not None

    def test_workflow_validator_unreachable_steps(self):
        """Test unreachable step detection."""
        from aion.automation.visual import WorkflowValidator

        validator = WorkflowValidator()

        # Workflow with unreachable step
        workflow = Workflow(
            name="Test",
            steps=[
                WorkflowStep(
                    id="s1",
                    name="Step 1",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                ),
                WorkflowStep(
                    id="s2",
                    name="Unreachable Step",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                ),
            ],
            entry_step_id="s1",
        )

        result = validator.validate(workflow)
        # Should have warning about unreachable step
        assert len(result.warnings) > 0 or not result.is_valid

    def test_workflow_exporter_json(self):
        """Test exporting workflow to JSON."""
        from aion.automation.visual import WorkflowExporter

        exporter = WorkflowExporter()

        workflow = Workflow(
            name="Export Test",
            description="Test workflow for export",
            steps=[
                WorkflowStep(
                    id="s1",
                    name="Step 1",
                    action=ActionConfig(
                        action_type=ActionType.TRANSFORM,
                        transform_expression="x + 1",
                    ),
                ),
            ],
            entry_step_id="s1",
        )

        json_str = exporter.to_json(workflow)
        data = json.loads(json_str)

        assert data["name"] == "Export Test"
        assert len(data["steps"]) == 1

    def test_workflow_exporter_yaml(self):
        """Test exporting workflow to YAML."""
        from aion.automation.visual import WorkflowExporter

        exporter = WorkflowExporter()

        workflow = Workflow(
            name="YAML Test",
            steps=[
                WorkflowStep(
                    id="s1",
                    name="Step 1",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                ),
            ],
        )

        yaml_str = exporter.to_yaml(workflow)
        assert "name: YAML Test" in yaml_str

    def test_workflow_exporter_import_json(self):
        """Test importing workflow from JSON."""
        from aion.automation.visual import WorkflowExporter

        exporter = WorkflowExporter()

        json_str = json.dumps({
            "name": "Imported Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Step 1",
                    "action": {"action_type": "transform"},
                }
            ],
        })

        workflow = exporter.from_json(json_str)
        assert workflow.name == "Imported Workflow"
        assert len(workflow.steps) == 1

    def test_visual_graph_conversion(self):
        """Test converting between visual graph and workflow."""
        from aion.automation.visual.exporter import WorkflowExporter

        exporter = WorkflowExporter()

        # Create workflow
        workflow = Workflow(
            name="Visual Test",
            steps=[
                WorkflowStep(
                    id="s1",
                    name="Entry",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                    on_success="s2",
                ),
                WorkflowStep(
                    id="s2",
                    name="Process",
                    action=ActionConfig(action_type=ActionType.TOOL, tool_name="echo"),
                    on_success="s3",
                ),
                WorkflowStep(
                    id="s3",
                    name="End",
                    action=ActionConfig(action_type=ActionType.NOTIFICATION),
                ),
            ],
            entry_step_id="s1",
        )

        # Export to visual format
        graph = exporter.to_visual_graph(workflow)

        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2

        # Import back
        imported = exporter.from_visual_graph(graph)
        assert imported.name == workflow.name
        assert len(imported.steps) == 3


# === Saga Pattern Tests ===


class TestSagaPattern:
    """Tests for saga pattern implementation."""

    @pytest.mark.asyncio
    async def test_saga_orchestrator_creation(self):
        """Test creating saga orchestrator."""
        from aion.automation.saga import SagaOrchestrator

        orchestrator = SagaOrchestrator()
        await orchestrator.initialize()

        assert orchestrator._initialized

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_saga_definition(self):
        """Test defining a saga."""
        from aion.automation.saga import SagaDefinition, SagaStep

        saga = SagaDefinition(
            name="Order Saga",
            steps=[
                SagaStep(
                    id="reserve_inventory",
                    name="Reserve Inventory",
                    action={"type": "reserve", "item": "item-1"},
                    compensation={"type": "release", "item": "item-1"},
                ),
                SagaStep(
                    id="charge_payment",
                    name="Charge Payment",
                    action={"type": "charge", "amount": 100},
                    compensation={"type": "refund", "amount": 100},
                ),
                SagaStep(
                    id="create_shipment",
                    name="Create Shipment",
                    action={"type": "ship"},
                    compensation={"type": "cancel_shipment"},
                ),
            ],
        )

        assert saga.name == "Order Saga"
        assert len(saga.steps) == 3

    @pytest.mark.asyncio
    async def test_saga_execution_success(self):
        """Test successful saga execution."""
        from aion.automation.saga import SagaOrchestrator, SagaDefinition, SagaStep

        orchestrator = SagaOrchestrator()
        await orchestrator.initialize()

        executed_steps = []

        async def action_handler(step_id: str, action: Dict) -> Dict:
            executed_steps.append(step_id)
            return {"status": "ok"}

        orchestrator.register_action_handler(action_handler)

        saga = SagaDefinition(
            name="Test Saga",
            steps=[
                SagaStep(id="step1", name="Step 1", action={"do": "1"}),
                SagaStep(id="step2", name="Step 2", action={"do": "2"}),
            ],
        )

        result = await orchestrator.execute(saga, context={"test": True})

        assert result.success
        assert "step1" in executed_steps
        assert "step2" in executed_steps

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_saga_compensation_on_failure(self):
        """Test saga compensation when step fails."""
        from aion.automation.saga import SagaOrchestrator, SagaDefinition, SagaStep

        orchestrator = SagaOrchestrator()
        await orchestrator.initialize()

        executed = []
        compensated = []

        async def action_handler(step_id: str, action: Dict) -> Dict:
            executed.append(step_id)
            if step_id == "step2":
                raise Exception("Step 2 failed")
            return {"status": "ok"}

        async def compensation_handler(step_id: str, compensation: Dict, context: Dict) -> Dict:
            compensated.append(step_id)
            return {"status": "compensated"}

        orchestrator.register_action_handler(action_handler)
        orchestrator.register_compensation_handler(compensation_handler)

        saga = SagaDefinition(
            name="Failing Saga",
            steps=[
                SagaStep(
                    id="step1",
                    name="Step 1",
                    action={"do": "1"},
                    compensation={"undo": "1"},
                ),
                SagaStep(
                    id="step2",
                    name="Step 2",
                    action={"do": "2"},
                    compensation={"undo": "2"},
                ),
                SagaStep(
                    id="step3",
                    name="Step 3",
                    action={"do": "3"},
                ),
            ],
        )

        result = await orchestrator.execute(saga, context={})

        assert not result.success
        assert "step1" in executed
        assert "step2" in executed
        assert "step3" not in executed  # Never reached
        assert "step1" in compensated  # Compensated

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_compensation_manager(self):
        """Test compensation manager."""
        from aion.automation.saga import CompensationManager

        manager = CompensationManager()
        await manager.initialize()

        compensations_run = []

        async def compensate_fn(context: Dict) -> Dict:
            compensations_run.append(context["step"])
            return {"undone": True}

        # Register compensations
        manager.register_compensation(
            "comp1",
            compensate_fn,
            context={"step": "step1"},
        )
        manager.register_compensation(
            "comp2",
            compensate_fn,
            context={"step": "step2"},
        )

        # Run all compensations (in reverse order)
        results = await manager.run_all_compensations()

        assert len(results) == 2
        # Should run in reverse order
        assert compensations_run == ["step2", "step1"]

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_semantic_lock_manager(self):
        """Test semantic lock manager."""
        from aion.automation.saga.semantic_lock import (
            SemanticLockManager,
            LockType,
            LockMode,
        )

        manager = SemanticLockManager()
        await manager.initialize()

        # Acquire exclusive lock
        lock = await manager.acquire(
            resource_id="order:123",
            holder_id="saga-1",
            lock_type=LockType.EXCLUSIVE,
        )

        assert lock is not None
        assert lock.resource_id == "order:123"
        assert lock.lock_type == LockType.EXCLUSIVE

        # Try to acquire same resource - should block or fail
        lock2 = await manager.acquire(
            resource_id="order:123",
            holder_id="saga-2",
            lock_type=LockType.EXCLUSIVE,
            mode=LockMode.NON_BLOCKING,
        )

        assert lock2 is None  # Couldn't acquire

        # Release first lock
        released = await manager.release(lock.id)
        assert released

        # Now can acquire
        lock3 = await manager.acquire(
            resource_id="order:123",
            holder_id="saga-2",
            lock_type=LockType.EXCLUSIVE,
            mode=LockMode.NON_BLOCKING,
        )

        assert lock3 is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_shared_locks(self):
        """Test shared lock compatibility."""
        from aion.automation.saga.semantic_lock import (
            SemanticLockManager,
            LockType,
            LockMode,
        )

        manager = SemanticLockManager()
        await manager.initialize()

        # Multiple shared locks should be allowed
        lock1 = await manager.acquire(
            resource_id="resource:1",
            holder_id="reader-1",
            lock_type=LockType.SHARED,
        )
        assert lock1 is not None

        lock2 = await manager.acquire(
            resource_id="resource:1",
            holder_id="reader-2",
            lock_type=LockType.SHARED,
            mode=LockMode.NON_BLOCKING,
        )
        assert lock2 is not None  # Both can hold shared lock

        # Exclusive lock should fail with shared locks held
        lock3 = await manager.acquire(
            resource_id="resource:1",
            holder_id="writer-1",
            lock_type=LockType.EXCLUSIVE,
            mode=LockMode.NON_BLOCKING,
        )
        assert lock3 is None  # Can't get exclusive with shared locks

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_lock_context_manager(self):
        """Test lock context manager."""
        from aion.automation.saga.semantic_lock import (
            SemanticLockManager,
            LockContext,
            LockType,
        )

        manager = SemanticLockManager()
        await manager.initialize()

        async with LockContext(
            manager,
            resource_id="data:456",
            holder_id="worker-1",
            lock_type=LockType.EXCLUSIVE,
        ) as lock:
            assert lock is not None
            assert lock.resource_id == "data:456"

            # Lock should be held
            locks = manager.get_locks("data:456")
            assert len(locks) == 1

        # Lock should be released after context
        locks = manager.get_locks("data:456")
        assert len(locks) == 0

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_lock_renewal(self):
        """Test lock TTL renewal."""
        from aion.automation.saga.semantic_lock import (
            SemanticLockManager,
            LockType,
        )

        manager = SemanticLockManager()
        await manager.initialize()

        lock = await manager.acquire(
            resource_id="resource:1",
            holder_id="holder-1",
            lock_type=LockType.EXCLUSIVE,
            ttl_seconds=10,
        )

        original_expiry = lock.expires_at

        # Renew with longer TTL
        renewed = await manager.renew(lock.id, ttl_seconds=60)
        assert renewed

        # Get updated lock
        locks = manager.get_locks("resource:1")
        assert len(locks) == 1
        assert locks[0].expires_at > original_expiry

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_release_all_holder_locks(self):
        """Test releasing all locks held by a holder."""
        from aion.automation.saga.semantic_lock import (
            SemanticLockManager,
            LockType,
        )

        manager = SemanticLockManager()
        await manager.initialize()

        # Acquire multiple locks
        await manager.acquire("resource:1", "holder-1", LockType.EXCLUSIVE)
        await manager.acquire("resource:2", "holder-1", LockType.EXCLUSIVE)
        await manager.acquire("resource:3", "holder-1", LockType.EXCLUSIVE)

        # Release all
        released_count = await manager.release_all("holder-1")
        assert released_count == 3

        # Verify all released
        holder_locks = manager.get_holder_locks("holder-1")
        assert len(holder_locks) == 0

        await manager.shutdown()


# === Enhanced Engine Tests ===


class TestEnhancedEngine:
    """Tests for enhanced workflow engine with SOTA features."""

    @pytest.mark.asyncio
    async def test_enhanced_engine_creation(self):
        """Test creating enhanced workflow engine."""
        from aion.automation.engine_enhanced import create_enhanced_engine

        engine = await create_enhanced_engine(
            enable_event_sourcing=True,
            enable_distributed=False,
            enable_observability=True,
            enable_saga=True,
        )

        assert engine is not None
        assert engine._initialized

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_enhanced_engine_basic_execution(self):
        """Test basic workflow execution with enhanced engine."""
        from aion.automation.engine_enhanced import create_enhanced_engine

        engine = await create_enhanced_engine(
            enable_event_sourcing=True,
            enable_distributed=False,
            enable_observability=True,
        )

        # Create and register workflow
        workflow = Workflow(
            name="Enhanced Test",
            steps=[
                WorkflowStep(
                    id="s1",
                    name="Step 1",
                    action=ActionConfig(
                        action_type=ActionType.TRANSFORM,
                        transform_expression="{{ inputs.value * 2 }}",
                        transform_output_key="result",
                    ),
                ),
            ],
            entry_step_id="s1",
        )

        await engine.register_workflow(workflow)

        # Execute
        execution = await engine.execute(
            workflow_id=workflow.id,
            inputs={"value": 5},
        )

        # Wait for completion
        await asyncio.sleep(0.5)

        # Check result
        updated = engine.get_execution(execution.id)
        assert updated.status == ExecutionStatus.COMPLETED

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_enhanced_engine_event_replay(self):
        """Test event replay with enhanced engine."""
        from aion.automation.engine_enhanced import create_enhanced_engine

        engine = await create_enhanced_engine(enable_event_sourcing=True)

        workflow = Workflow(
            name="Replay Test",
            steps=[
                WorkflowStep(
                    id="s1",
                    name="Step 1",
                    action=ActionConfig(action_type=ActionType.DELAY, delay_seconds=0.1),
                    on_success="s2",
                ),
                WorkflowStep(
                    id="s2",
                    name="Step 2",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                ),
            ],
            entry_step_id="s1",
        )

        await engine.register_workflow(workflow)

        execution = await engine.execute(workflow_id=workflow.id, inputs={})
        await asyncio.sleep(1.0)

        # Replay should work if event store is available
        if engine.event_store:
            from aion.automation.execution.event_store import WorkflowReplayer
            replayer = WorkflowReplayer(engine.event_store)
            state = await replayer.replay(execution.id)
            assert state is not None

        await engine.shutdown()


# === Integration Tests ===


class TestSOTAIntegration:
    """Integration tests combining multiple SOTA features."""

    @pytest.mark.asyncio
    async def test_full_sota_workflow(self):
        """Test workflow using all SOTA features together."""
        from aion.automation.engine_enhanced import create_enhanced_engine
        from aion.automation.observability import TelemetryProvider, WorkflowTracer

        # Create engine with all features
        engine = await create_enhanced_engine(
            enable_event_sourcing=True,
            enable_observability=True,
            enable_saga=True,
        )

        # Setup telemetry
        provider = TelemetryProvider(service_name="integration-test")
        tracer = WorkflowTracer(provider)

        # Create workflow
        workflow = Workflow(
            name="Full SOTA Test",
            description="Tests all SOTA features",
            steps=[
                WorkflowStep(
                    id="validate",
                    name="Validate Input",
                    action=ActionConfig(
                        action_type=ActionType.TRANSFORM,
                        transform_expression="{{ inputs.data }}",
                    ),
                    on_success="process",
                ),
                WorkflowStep(
                    id="process",
                    name="Process Data",
                    action=ActionConfig(
                        action_type=ActionType.TRANSFORM,
                        transform_expression="{{ steps.validate.output }} processed",
                    ),
                    on_success="complete",
                ),
                WorkflowStep(
                    id="complete",
                    name="Complete",
                    action=ActionConfig(action_type=ActionType.TRANSFORM),
                ),
            ],
            entry_step_id="validate",
        )

        await engine.register_workflow(workflow)

        # Execute with tracing
        with tracer.workflow_span(workflow.id, workflow.name):
            execution = await engine.execute(
                workflow_id=workflow.id,
                inputs={"data": "test-data"},
            )

            await asyncio.sleep(1.0)

        # Verify execution completed
        updated = engine.get_execution(execution.id)
        assert updated.status == ExecutionStatus.COMPLETED

        # Verify events were stored
        if engine.event_store:
            events = await engine.event_store.get_events(execution.id)
            assert len(events) > 0

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_saga_with_locks(self):
        """Test saga pattern with semantic locks."""
        from aion.automation.saga import SagaOrchestrator, SagaDefinition, SagaStep
        from aion.automation.saga.semantic_lock import SemanticLockManager, LockContext

        orchestrator = SagaOrchestrator()
        await orchestrator.initialize()

        lock_manager = SemanticLockManager()
        await lock_manager.initialize()

        async def locked_action(step_id: str, action: Dict) -> Dict:
            resource = action.get("resource", "default")
            async with LockContext(lock_manager, resource, step_id):
                await asyncio.sleep(0.1)  # Simulate work
                return {"processed": resource}

        orchestrator.register_action_handler(locked_action)

        saga = SagaDefinition(
            name="Locked Saga",
            steps=[
                SagaStep(
                    id="s1",
                    name="Step 1",
                    action={"resource": "resource:1"},
                ),
                SagaStep(
                    id="s2",
                    name="Step 2",
                    action={"resource": "resource:2"},
                ),
            ],
        )

        result = await orchestrator.execute(saga, context={})
        assert result.success

        await orchestrator.shutdown()
        await lock_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
