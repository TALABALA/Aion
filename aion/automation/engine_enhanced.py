"""
AION Enhanced Workflow Engine

Production-grade workflow engine with SOTA features:
- Event-sourced execution with replay
- Distributed task queue
- OpenTelemetry observability
- Saga pattern with compensation
- Visual workflow support
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import structlog

from aion.automation.types import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    StepResult,
    ExecutionStatus,
    WorkflowStatus,
)
from aion.automation.engine import WorkflowEngine
from aion.automation.registry import WorkflowRegistry
from aion.automation.triggers.manager import TriggerManager

# SOTA imports
try:
    from aion.automation.execution.event_store import (
        EventStore,
        EventType,
        WorkflowReplayer,
        FileEventStore,
        RedisEventStore,
    )
    EVENT_SOURCING_AVAILABLE = True
except ImportError:
    EVENT_SOURCING_AVAILABLE = False

try:
    from aion.automation.distributed import (
        TaskQueue,
        Worker,
        WorkerPool,
        DistributedScheduler,
        create_backend,
    )
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    from aion.automation.observability import (
        TelemetryProvider,
        TracingConfig,
        MetricsConfig,
        configure_telemetry,
        WorkflowTracer,
        WorkflowMetrics,
        get_workflow_tracer,
        get_workflow_metrics,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

try:
    from aion.automation.saga import (
        SagaOrchestrator,
        SagaDefinition,
        CompensationManager,
        SemanticLockManager,
    )
    SAGA_AVAILABLE = True
except ImportError:
    SAGA_AVAILABLE = False

try:
    from aion.automation.visual import (
        WorkflowValidator,
        WorkflowExporter,
        validate_workflow,
    )
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False

logger = structlog.get_logger(__name__)


class EnhancedWorkflowEngine(WorkflowEngine):
    """
    Enhanced workflow engine with SOTA features.

    Extends the base WorkflowEngine with:
    - Event sourcing for durability and replay
    - Distributed execution via task queue
    - OpenTelemetry tracing and metrics
    - Saga pattern for distributed transactions
    - Visual workflow support
    """

    def __init__(
        self,
        registry: Optional[WorkflowRegistry] = None,
        trigger_manager: Optional[TriggerManager] = None,
        max_concurrent_executions: int = 100,
        # Event sourcing config
        enable_event_sourcing: bool = True,
        event_store_backend: str = "file",  # file, redis, memory
        event_store_path: str = "./data/events",
        event_store_redis_url: Optional[str] = None,
        # Distributed config
        enable_distributed: bool = False,
        queue_backend: str = "memory",  # memory, redis, rabbitmq
        queue_redis_url: Optional[str] = None,
        queue_rabbitmq_url: Optional[str] = None,
        worker_count: int = 4,
        # Observability config
        enable_observability: bool = True,
        tracing_config: Optional["TracingConfig"] = None,
        metrics_config: Optional["MetricsConfig"] = None,
        # Saga config
        enable_sagas: bool = True,
    ):
        super().__init__(registry, trigger_manager, max_concurrent_executions)

        # Configuration
        self._enable_event_sourcing = enable_event_sourcing and EVENT_SOURCING_AVAILABLE
        self._enable_distributed = enable_distributed and DISTRIBUTED_AVAILABLE
        self._enable_observability = enable_observability and OBSERVABILITY_AVAILABLE
        self._enable_sagas = enable_sagas and SAGA_AVAILABLE

        # Event store config
        self._event_store_backend = event_store_backend
        self._event_store_path = event_store_path
        self._event_store_redis_url = event_store_redis_url

        # Distributed config
        self._queue_backend = queue_backend
        self._queue_redis_url = queue_redis_url
        self._queue_rabbitmq_url = queue_rabbitmq_url
        self._worker_count = worker_count

        # Observability config
        self._tracing_config = tracing_config
        self._metrics_config = metrics_config

        # SOTA components (initialized in initialize())
        self._event_store: Optional["EventStore"] = None
        self._replayer: Optional["WorkflowReplayer"] = None
        self._task_queue: Optional["TaskQueue"] = None
        self._worker_pool: Optional["WorkerPool"] = None
        self._telemetry: Optional["TelemetryProvider"] = None
        self._tracer: Optional["WorkflowTracer"] = None
        self._metrics: Optional["WorkflowMetrics"] = None
        self._saga_orchestrator: Optional["SagaOrchestrator"] = None
        self._compensation_manager: Optional["CompensationManager"] = None
        self._lock_manager: Optional["SemanticLockManager"] = None
        self._validator: Optional["WorkflowValidator"] = None

    async def initialize(self) -> None:
        """Initialize the enhanced workflow engine with all SOTA components."""
        if self._initialized:
            return

        logger.info("Initializing Enhanced Workflow Engine")

        # Initialize base engine
        await super().initialize()

        # Initialize SOTA components
        await self._initialize_event_sourcing()
        await self._initialize_distributed()
        await self._initialize_observability()
        await self._initialize_sagas()
        await self._initialize_visual()

        # Log capability summary
        logger.info(
            "Enhanced Workflow Engine initialized",
            event_sourcing=self._enable_event_sourcing,
            distributed=self._enable_distributed,
            observability=self._enable_observability,
            sagas=self._enable_sagas,
        )

    async def _initialize_event_sourcing(self) -> None:
        """Initialize event sourcing components."""
        if not self._enable_event_sourcing:
            return

        try:
            from aion.automation.execution.event_store import (
                EventStore,
                FileEventStore,
                RedisEventStore,
                InMemoryEventStore,
                WorkflowReplayer,
            )

            # Create backend
            if self._event_store_backend == "redis" and self._event_store_redis_url:
                backend = RedisEventStore(redis_url=self._event_store_redis_url)
            elif self._event_store_backend == "file":
                from pathlib import Path
                backend = FileEventStore(Path(self._event_store_path))
            else:
                backend = InMemoryEventStore()

            self._event_store = EventStore(backend=backend)
            self._replayer = WorkflowReplayer(self._event_store)

            logger.info(f"Event sourcing initialized with {self._event_store_backend} backend")

        except Exception as e:
            logger.warning(f"Failed to initialize event sourcing: {e}")
            self._enable_event_sourcing = False

    async def _initialize_distributed(self) -> None:
        """Initialize distributed task queue."""
        if not self._enable_distributed:
            return

        try:
            from aion.automation.distributed import (
                TaskQueue,
                WorkerPool,
                create_backend,
            )

            # Create queue backend
            if self._queue_backend == "redis" and self._queue_redis_url:
                backend = create_backend("redis", redis_url=self._queue_redis_url)
            elif self._queue_backend == "rabbitmq" and self._queue_rabbitmq_url:
                backend = create_backend("rabbitmq", amqp_url=self._queue_rabbitmq_url)
            else:
                backend = create_backend("memory")

            self._task_queue = TaskQueue(backend=backend)
            await self._task_queue.initialize()

            # Create worker pool
            self._worker_pool = WorkerPool(
                queue=self._task_queue,
                min_workers=1,
                max_workers=self._worker_count,
            )

            # Register step handler
            self._worker_pool.register_handler("execute_step", self._execute_step_task)
            await self._worker_pool.start()

            logger.info(f"Distributed queue initialized with {self._queue_backend} backend")

        except Exception as e:
            logger.warning(f"Failed to initialize distributed queue: {e}")
            self._enable_distributed = False

    async def _initialize_observability(self) -> None:
        """Initialize OpenTelemetry observability."""
        if not self._enable_observability:
            return

        try:
            from aion.automation.observability import (
                TelemetryProvider,
                TracingConfig,
                MetricsConfig,
                WorkflowTracer,
                WorkflowMetrics,
            )

            tracing_config = self._tracing_config or TracingConfig(
                service_name="aion-automation",
                exporter="console" if not self._tracing_config else self._tracing_config.exporter,
            )

            metrics_config = self._metrics_config or MetricsConfig(
                service_name="aion-automation",
            )

            self._telemetry = TelemetryProvider(
                tracing_config=tracing_config,
                metrics_config=metrics_config,
            )
            self._telemetry.initialize()

            self._tracer = WorkflowTracer(self._telemetry.tracer)
            self._metrics = WorkflowMetrics()

            logger.info("Observability initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize observability: {e}")
            self._enable_observability = False

    async def _initialize_sagas(self) -> None:
        """Initialize saga pattern components."""
        if not self._enable_sagas:
            return

        try:
            from aion.automation.saga import (
                SagaOrchestrator,
                CompensationManager,
                SemanticLockManager,
            )

            self._saga_orchestrator = SagaOrchestrator(
                event_store=self._event_store if self._enable_event_sourcing else None,
            )
            await self._saga_orchestrator.initialize()

            self._compensation_manager = CompensationManager()
            await self._compensation_manager.initialize()

            self._lock_manager = SemanticLockManager()
            await self._lock_manager.initialize()

            logger.info("Saga pattern initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize sagas: {e}")
            self._enable_sagas = False

    async def _initialize_visual(self) -> None:
        """Initialize visual workflow components."""
        if not VISUAL_AVAILABLE:
            return

        try:
            from aion.automation.visual import WorkflowValidator
            self._validator = WorkflowValidator()
            logger.info("Visual workflow support initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize visual support: {e}")

    async def shutdown(self) -> None:
        """Shutdown the enhanced workflow engine."""
        logger.info("Shutting down Enhanced Workflow Engine")

        # Shutdown SOTA components
        if self._worker_pool:
            await self._worker_pool.stop()
        if self._task_queue:
            await self._task_queue.shutdown()
        if self._saga_orchestrator:
            await self._saga_orchestrator.shutdown()
        if self._compensation_manager:
            await self._compensation_manager.shutdown()
        if self._lock_manager:
            await self._lock_manager.shutdown()
        if self._telemetry:
            self._telemetry.shutdown()

        # Shutdown base engine
        await super().shutdown()

        logger.info("Enhanced Workflow Engine shutdown complete")

    # === Enhanced Execution with Event Sourcing ===

    async def execute(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        triggered_by: Optional[str] = None,
    ) -> WorkflowExecution:
        """
        Execute a workflow with event sourcing and observability.

        Extends the base execute method with:
        - Event recording
        - Distributed tracing
        - Metrics collection
        - Saga support
        """
        # Start tracing
        if self._tracer:
            trace_context = self._tracer.trace_workflow_execution(
                execution_id=execution_id or "pending",
                workflow_id=workflow_id,
                workflow_name=workflow_id,  # Will be updated
                inputs=inputs,
            )
            trace_context.__enter__()

        try:
            # Record execution start event
            if self._event_store and execution_id:
                await self._event_store.append(
                    execution_id,
                    EventType.WORKFLOW_STARTED,
                    {
                        "workflow_id": workflow_id,
                        "inputs": inputs,
                        "triggered_by": triggered_by,
                    },
                )

            # Record metric
            if self._metrics:
                self._metrics.workflow_started(
                    workflow_id=workflow_id,
                    workflow_name=workflow_id,
                    trigger_type=triggered_by,
                )

            # Execute workflow
            execution = await super().execute(
                workflow_id=workflow_id,
                inputs=inputs,
                execution_id=execution_id,
                triggered_by=triggered_by,
            )

            # Record completion event
            if self._event_store:
                if execution.status == ExecutionStatus.COMPLETED:
                    await self._event_store.append(
                        execution.id,
                        EventType.WORKFLOW_COMPLETED,
                        {"result": execution.result},
                    )
                elif execution.status == ExecutionStatus.FAILED:
                    await self._event_store.append(
                        execution.id,
                        EventType.WORKFLOW_FAILED,
                        {"error": execution.error},
                    )

            # Record metrics
            if self._metrics and execution.started_at:
                duration = (datetime.now() - execution.started_at).total_seconds()
                if execution.status == ExecutionStatus.COMPLETED:
                    self._metrics.workflow_completed(
                        workflow_id=workflow_id,
                        workflow_name=workflow_id,
                        duration_seconds=duration,
                    )
                elif execution.status == ExecutionStatus.FAILED:
                    self._metrics.workflow_failed(
                        workflow_id=workflow_id,
                        workflow_name=workflow_id,
                        error_type=type(execution.error).__name__ if execution.error else "Unknown",
                        duration_seconds=duration,
                    )

            return execution

        finally:
            if self._tracer:
                trace_context.__exit__(None, None, None)

    async def _execute_step_with_saga(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        context: "ExecutionContext",
    ) -> StepResult:
        """Execute a step with saga compensation support."""
        # Register compensation before execution
        if self._compensation_manager and step.compensation_action:
            self._compensation_manager.register_compensation(
                execution_id=execution.id,
                step_id=step.id,
                name=f"Compensate {step.name}",
                action=step.compensation_action,
                original_context=context.to_dict(),
            )

        # Execute step with tracing
        if self._tracer:
            with self._tracer.trace_step(
                execution_id=execution.id,
                step_id=step.id,
                step_name=step.name,
                step_type=step.action.type.value if step.action else "unknown",
            ):
                return await self._execute_step(step, execution, context)
        else:
            return await self._execute_step(step, execution, context)

    async def _execute_step_task(self, payload: Dict[str, Any]) -> Any:
        """Task handler for distributed step execution."""
        step_id = payload.get("step_id")
        execution_id = payload.get("execution_id")
        context_data = payload.get("context", {})

        execution = self._executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        workflow = await self.registry.get(execution.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {execution.workflow_id}")

        step = workflow.get_step(step_id)
        if not step:
            raise ValueError(f"Step not found: {step_id}")

        from aion.automation.execution.context import ExecutionContext
        context = ExecutionContext(
            execution_id=execution_id,
            workflow_id=workflow.id,
            inputs=execution.inputs,
        )
        context._data = context_data

        return await self._execute_step(step, execution, context)

    # === Replay Capabilities ===

    async def replay_execution(
        self,
        execution_id: str,
        to_sequence: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Replay an execution from event store.

        Args:
            execution_id: The execution to replay
            to_sequence: Optional sequence to replay up to (for time-travel)

        Returns:
            Reconstructed execution state
        """
        if not self._replayer:
            raise RuntimeError("Event sourcing not enabled")

        return await self._replayer.replay(execution_id, to_sequence)

    async def get_execution_timeline(
        self,
        execution_id: str,
    ) -> List[Dict[str, Any]]:
        """Get the event timeline for an execution."""
        if not self._event_store:
            raise RuntimeError("Event sourcing not enabled")

        return await self._event_store.get_execution_timeline(execution_id)

    # === Saga Operations ===

    async def execute_saga(
        self,
        saga_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> "SagaState":
        """Execute a registered saga."""
        if not self._saga_orchestrator:
            raise RuntimeError("Saga support not enabled")

        return await self._saga_orchestrator.execute(saga_id, context)

    def register_saga(self, definition: "SagaDefinition") -> None:
        """Register a saga definition."""
        if not self._saga_orchestrator:
            raise RuntimeError("Saga support not enabled")

        self._saga_orchestrator.register_saga(definition)

    # === Lock Operations ===

    async def acquire_lock(
        self,
        resource_id: str,
        holder_id: str,
        lock_type: str = "exclusive",
        timeout_seconds: Optional[float] = None,
    ) -> Optional["Lock"]:
        """Acquire a semantic lock."""
        if not self._lock_manager:
            raise RuntimeError("Saga support not enabled")

        from aion.automation.saga import LockType, LockMode

        return await self._lock_manager.acquire(
            resource_id=resource_id,
            holder_id=holder_id,
            lock_type=LockType(lock_type),
            mode=LockMode.TIMEOUT if timeout_seconds else LockMode.BLOCKING,
            timeout_seconds=timeout_seconds,
        )

    async def release_lock(self, lock_id: str) -> bool:
        """Release a semantic lock."""
        if not self._lock_manager:
            raise RuntimeError("Saga support not enabled")

        return await self._lock_manager.release(lock_id)

    # === Validation ===

    def validate_workflow(self, workflow: Workflow) -> "ValidationResult":
        """Validate a workflow definition."""
        if not self._validator:
            # Fall back to basic validation
            errors = workflow.validate()
            return {"valid": len(errors) == 0, "errors": errors, "warnings": []}

        return self._validator.validate(workflow)

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        stats = {
            "engine": {
                "initialized": self._initialized,
                "active_executions": len(self._executions),
            },
            "features": {
                "event_sourcing": self._enable_event_sourcing,
                "distributed": self._enable_distributed,
                "observability": self._enable_observability,
                "sagas": self._enable_sagas,
            },
        }

        if self._task_queue:
            stats["task_queue"] = asyncio.create_task(
                self._task_queue.get_queue_stats("default")
            )

        if self._worker_pool:
            stats["workers"] = self._worker_pool.get_stats()

        if self._saga_orchestrator:
            stats["sagas"] = self._saga_orchestrator.get_stats()

        if self._lock_manager:
            stats["locks"] = self._lock_manager.get_stats()

        if self._compensation_manager:
            stats["compensations"] = self._compensation_manager.get_stats()

        return stats


# Factory function
def create_enhanced_engine(
    config: Optional[Dict[str, Any]] = None,
) -> EnhancedWorkflowEngine:
    """
    Create an enhanced workflow engine with configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured EnhancedWorkflowEngine
    """
    config = config or {}

    return EnhancedWorkflowEngine(
        # Event sourcing
        enable_event_sourcing=config.get("enable_event_sourcing", True),
        event_store_backend=config.get("event_store_backend", "file"),
        event_store_path=config.get("event_store_path", "./data/events"),
        event_store_redis_url=config.get("event_store_redis_url"),
        # Distributed
        enable_distributed=config.get("enable_distributed", False),
        queue_backend=config.get("queue_backend", "memory"),
        queue_redis_url=config.get("queue_redis_url"),
        queue_rabbitmq_url=config.get("queue_rabbitmq_url"),
        worker_count=config.get("worker_count", 4),
        # Observability
        enable_observability=config.get("enable_observability", True),
        # Sagas
        enable_sagas=config.get("enable_sagas", True),
        # Concurrency
        max_concurrent_executions=config.get("max_concurrent_executions", 100),
    )
