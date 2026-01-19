"""
AION Kernel - The Core Cognitive Architecture

The kernel is the central orchestrator that:
- Manages all subsystems (planning, memory, tools, evolution, vision)
- Handles request routing and execution
- Maintains system state and health
- Provides the unified API for external interactions
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import structlog

from aion.core.config import AIONConfig, get_config
from aion.core.security import SecurityManager, RiskLevel
from aion.core.llm import LLMAdapter, LLMConfig, LLMProvider, Message

# Process manager imports (conditional to avoid import errors)
try:
    from aion.systems.process import (
        ProcessSupervisor,
        EventBus,
        TaskScheduler,
        WorkerPool,
        ResourceManager,
        AgentConfig,
        ProcessPriority,
        RestartPolicy,
        ResourceLimits,
    )
    PROCESS_MANAGER_AVAILABLE = True
except ImportError:
    PROCESS_MANAGER_AVAILABLE = False

logger = structlog.get_logger(__name__)


class SystemStatus(str, Enum):
    """Status of AION systems."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class SystemHealth:
    """Health status of a subsystem."""
    name: str
    status: SystemStatus
    last_check: datetime
    latency_ms: float = 0.0
    error_count: int = 0
    message: Optional[str] = None


@dataclass
class ExecutionContext:
    """Context for a single execution request."""
    request_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of an execution request."""
    request_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    steps_executed: int = 0
    memory_entries_created: int = 0
    tools_invoked: list[str] = field(default_factory=list)


class AIONKernel:
    """
    AION Kernel - The Central Cognitive Architecture

    The kernel coordinates all cognitive subsystems and provides
    the main interface for interacting with AION.
    """

    def __init__(self, config: Optional[AIONConfig] = None):
        self.config = config or get_config()
        self.instance_id = self.config.instance_id

        # Core components
        self._security: Optional[SecurityManager] = None
        self._llm: Optional[LLMAdapter] = None

        # Subsystems (initialized lazily)
        self._planning_graph = None
        self._memory_system = None
        self._tool_orchestrator = None
        self._evolution_engine = None
        self._visual_cortex = None

        # Process Manager components
        self._event_bus = None
        self._supervisor = None
        self._scheduler = None
        self._worker_pool = None
        self._resource_manager = None

        # State
        self._status = SystemStatus.INITIALIZING
        self._health: dict[str, SystemHealth] = {}
        self._start_time: Optional[datetime] = None
        self._request_count = 0
        self._active_requests: dict[str, ExecutionContext] = {}

        # Locks
        self._init_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def initialize(self) -> None:
        """Initialize the kernel and all subsystems."""
        async with self._init_lock:
            if self._status == SystemStatus.READY:
                return

            logger.info("Initializing AION Kernel", instance_id=self.instance_id)
            self._start_time = datetime.now()

            try:
                # Ensure directories exist
                self.config.ensure_directories()

                # Initialize security manager
                self._security = SecurityManager(
                    require_approval_for_high_risk=self.config.security.require_approval_for_high_risk,
                    auto_approve_low_risk=self.config.security.auto_approve_low_risk,
                    approval_timeout=self.config.security.approval_timeout,
                    audit_all=self.config.security.audit_all_actions,
                    rate_limit=self.config.security.requests_per_minute / 60.0
                    if self.config.security.rate_limit_requests else None,
                )
                self._update_health("security", SystemStatus.READY)

                # Initialize LLM adapter
                llm_config = LLMConfig(
                    provider=LLMProvider(self.config.llm.provider),
                    model=self.config.llm.model,
                    api_key=self.config.get_llm_api_key(),
                    base_url=self.config.llm.base_url,
                    max_tokens=self.config.llm.max_tokens,
                    temperature=self.config.llm.temperature,
                    timeout=self.config.llm.timeout,
                    max_retries=self.config.llm.max_retries,
                )
                self._llm = LLMAdapter(llm_config)

                # Try to initialize LLM, but don't fail if no API key
                try:
                    await self._llm.initialize()
                    self._update_health("llm", SystemStatus.READY)
                except ValueError as e:
                    logger.warning("LLM initialization skipped", reason=str(e))
                    self._update_health("llm", SystemStatus.DEGRADED, str(e))

                # Initialize subsystems
                await self._initialize_subsystems()

                self._status = SystemStatus.READY
                logger.info(
                    "AION Kernel initialized successfully",
                    instance_id=self.instance_id,
                    status=self._status.value,
                )

            except Exception as e:
                self._status = SystemStatus.ERROR
                logger.error("Failed to initialize AION Kernel", error=str(e))
                raise

    async def _initialize_subsystems(self) -> None:
        """Initialize all cognitive subsystems."""
        # Import here to avoid circular imports
        from aion.systems.planning import PlanningGraph
        from aion.systems.memory import CognitiveMemorySystem
        from aion.systems.tools import ToolOrchestrator
        from aion.systems.evolution import SelfImprovementEngine
        from aion.systems.vision import VisualCortex

        # Initialize Planning Graph
        try:
            self._planning_graph = PlanningGraph(
                max_depth=self.config.planning.max_plan_depth,
                checkpoint_interval=self.config.planning.checkpoint_interval,
            )
            await self._planning_graph.initialize()
            self._update_health("planning", SystemStatus.READY)
        except Exception as e:
            logger.warning("Planning graph initialization failed", error=str(e))
            self._update_health("planning", SystemStatus.ERROR, str(e))

        # Initialize Memory System
        try:
            self._memory_system = CognitiveMemorySystem(
                embedding_model=self.config.memory.embedding_model,
                embedding_dim=self.config.memory.embedding_dimension,
                max_memories=self.config.memory.max_memories,
            )
            await self._memory_system.initialize()
            self._update_health("memory", SystemStatus.READY)
        except Exception as e:
            logger.warning("Memory system initialization failed", error=str(e))
            self._update_health("memory", SystemStatus.ERROR, str(e))

        # Initialize Tool Orchestrator
        try:
            self._tool_orchestrator = ToolOrchestrator(
                max_parallel=self.config.tools.max_parallel_tools,
                default_timeout=self.config.tools.tool_timeout,
            )
            await self._tool_orchestrator.initialize()
            self._update_health("tools", SystemStatus.READY)
        except Exception as e:
            logger.warning("Tool orchestrator initialization failed", error=str(e))
            self._update_health("tools", SystemStatus.ERROR, str(e))

        # Initialize Self-Improvement Engine
        if self.config.evolution.enable_self_improvement:
            try:
                self._evolution_engine = SelfImprovementEngine(
                    safety_threshold=self.config.evolution.safety_threshold,
                    require_approval=self.config.evolution.require_approval_for_changes,
                )
                await self._evolution_engine.initialize()
                self._update_health("evolution", SystemStatus.READY)
            except Exception as e:
                logger.warning("Evolution engine initialization failed", error=str(e))
                self._update_health("evolution", SystemStatus.ERROR, str(e))

        # Initialize Visual Cortex
        try:
            self._visual_cortex = VisualCortex(
                detection_model=self.config.vision.detection_model,
                enable_memory=self.config.vision.enable_visual_memory,
            )
            await self._visual_cortex.initialize()
            self._update_health("vision", SystemStatus.READY)
        except Exception as e:
            logger.warning("Visual cortex initialization failed", error=str(e))
            self._update_health("vision", SystemStatus.ERROR, str(e))

        # Initialize Process Manager (core OS layer)
        if PROCESS_MANAGER_AVAILABLE:
            await self._initialize_process_manager()

    async def _initialize_process_manager(self) -> None:
        """Initialize the Process & Agent Manager system."""
        proc_config = self.config.process

        try:
            # Initialize Event Bus
            self._event_bus = EventBus(
                max_history=proc_config.event_bus_max_history,
                max_dead_letters=proc_config.event_bus_max_dead_letters,
                default_ttl_seconds=proc_config.event_bus_default_ttl_seconds,
            )
            await self._event_bus.initialize()
            self._update_health("event_bus", SystemStatus.READY)

            # Initialize Resource Manager
            self._resource_manager = ResourceManager(
                system_limits=ResourceLimits(
                    max_memory_mb=proc_config.default_max_memory_mb,
                    max_tokens_per_minute=proc_config.default_max_tokens_per_minute,
                    max_tokens_total=proc_config.default_max_tokens_total,
                    max_runtime_seconds=proc_config.default_max_runtime_seconds,
                ),
                enable_memory_monitoring=proc_config.enable_resource_monitoring,
            )
            await self._resource_manager.initialize()

            # Initialize Process Supervisor
            self._supervisor = ProcessSupervisor(
                event_bus=self._event_bus,
                kernel=self,
                health_check_interval=proc_config.health_check_interval,
                max_processes=proc_config.max_processes,
                default_restart_delay=proc_config.default_restart_delay,
                default_max_restarts=proc_config.default_max_restarts,
                enable_resource_monitoring=proc_config.enable_resource_monitoring,
                zombie_timeout_seconds=proc_config.zombie_timeout_seconds,
            )
            await self._supervisor.initialize()
            self._update_health("supervisor", SystemStatus.READY)

            # Initialize Task Scheduler
            self._scheduler = TaskScheduler(
                supervisor=self._supervisor,
                event_bus=self._event_bus,
                check_interval=proc_config.scheduler_check_interval,
                max_concurrent_tasks=proc_config.max_concurrent_scheduled_tasks,
            )
            await self._scheduler.initialize()
            self._update_health("scheduler", SystemStatus.READY)

            # Initialize Worker Pool
            self._worker_pool = WorkerPool(
                event_bus=self._event_bus,
                min_workers=proc_config.worker_pool_min_workers,
                max_workers=proc_config.worker_pool_max_workers,
                max_queue_size=proc_config.worker_pool_max_queue_size,
                enable_auto_scaling=proc_config.worker_pool_enable_auto_scaling,
            )
            await self._worker_pool.initialize()
            self._update_health("worker_pool", SystemStatus.READY)

            # Start system agents
            await self._start_system_agents()

            logger.info("Process Manager initialized successfully")

        except Exception as e:
            logger.error("Process Manager initialization failed", error=str(e))
            self._update_health("process_manager", SystemStatus.ERROR, str(e))

    async def _start_system_agents(self) -> None:
        """Start built-in system agents."""
        if not self._supervisor:
            return

        proc_config = self.config.process

        # Health Monitor Agent
        if proc_config.enable_health_monitor:
            try:
                await self._supervisor.spawn_agent(AgentConfig(
                    name="system_health_monitor",
                    agent_class="health_monitor",
                    priority=ProcessPriority.CRITICAL,
                    restart_policy=RestartPolicy.ALWAYS,
                    metadata={
                        "check_interval": proc_config.health_monitor_interval,
                        "alert_thresholds": {
                            "memory_percent": proc_config.health_monitor_memory_threshold,
                            "cpu_percent": proc_config.health_monitor_cpu_threshold,
                        },
                    },
                ))
                logger.debug("Started health monitor agent")
            except Exception as e:
                logger.warning("Failed to start health monitor agent", error=str(e))

        # Garbage Collector Agent
        if proc_config.enable_garbage_collector:
            try:
                await self._supervisor.spawn_agent(AgentConfig(
                    name="system_gc",
                    agent_class="garbage_collector",
                    priority=ProcessPriority.LOW,
                    restart_policy=RestartPolicy.ALWAYS,
                    metadata={
                        "gc_interval": proc_config.gc_interval,
                        "max_completed_age": proc_config.gc_max_completed_age,
                    },
                ))
                logger.debug("Started garbage collector agent")
            except Exception as e:
                logger.warning("Failed to start garbage collector agent", error=str(e))

        # Metrics Collector Agent
        if proc_config.enable_metrics_collector:
            try:
                await self._supervisor.spawn_agent(AgentConfig(
                    name="system_metrics",
                    agent_class="metrics_collector",
                    priority=ProcessPriority.LOW,
                    restart_policy=RestartPolicy.ALWAYS,
                    metadata={
                        "collect_interval": proc_config.metrics_collect_interval,
                    },
                ))
                logger.debug("Started metrics collector agent")
            except Exception as e:
                logger.warning("Failed to start metrics collector agent", error=str(e))

        # Watchdog Agent
        if proc_config.enable_watchdog:
            try:
                await self._supervisor.spawn_agent(AgentConfig(
                    name="system_watchdog",
                    agent_class="watchdog",
                    priority=ProcessPriority.HIGH,
                    restart_policy=RestartPolicy.ALWAYS,
                    metadata={
                        "check_interval": proc_config.watchdog_check_interval,
                        "heartbeat_timeout": proc_config.watchdog_heartbeat_timeout,
                        "idle_timeout": proc_config.watchdog_idle_timeout,
                    },
                ))
                logger.debug("Started watchdog agent")
            except Exception as e:
                logger.warning("Failed to start watchdog agent", error=str(e))

    def _update_health(
        self,
        system: str,
        status: SystemStatus,
        message: Optional[str] = None,
    ) -> None:
        """Update health status for a system."""
        self._health[system] = SystemHealth(
            name=system,
            status=status,
            last_check=datetime.now(),
            message=message,
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown the kernel."""
        logger.info("Shutting down AION Kernel", instance_id=self.instance_id)
        self._status = SystemStatus.SHUTDOWN

        # Wait for active requests to complete (with timeout)
        if self._active_requests:
            logger.info(
                "Waiting for active requests",
                count=len(self._active_requests),
            )
            await asyncio.sleep(5)  # Give some time for requests to complete

        # Shutdown process manager first (graceful agent shutdown)
        if self._supervisor:
            await self._supervisor.shutdown()
        if self._scheduler:
            await self._scheduler.shutdown()
        if self._worker_pool:
            await self._worker_pool.shutdown()
        if self._resource_manager:
            await self._resource_manager.shutdown()
        if self._event_bus:
            await self._event_bus.shutdown()

        # Shutdown cognitive subsystems
        if self._llm:
            await self._llm.close()
        if self._planning_graph:
            await self._planning_graph.shutdown()
        if self._memory_system:
            await self._memory_system.shutdown()
        if self._tool_orchestrator:
            await self._tool_orchestrator.shutdown()
        if self._evolution_engine:
            await self._evolution_engine.shutdown()
        if self._visual_cortex:
            await self._visual_cortex.shutdown()

        logger.info("AION Kernel shutdown complete")

    # ==================== Core API ====================

    async def process_request(
        self,
        request: str,
        context: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Process a user request through the cognitive pipeline.

        This is the main entry point for all AION interactions.

        Args:
            request: The user's request/query
            context: Additional context for the request
            user_id: Optional user identifier
            session_id: Optional session identifier

        Returns:
            ExecutionResult with the outcome
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        exec_context = ExecutionContext(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.now(),
            metadata=context or {},
        )

        async with self._request_lock:
            self._active_requests[request_id] = exec_context
            self._request_count += 1

        try:
            # Security check
            authorized, reason = await self._security.authorize(
                operation="process_request",
                description=f"Process user request: {request[:100]}...",
                details={"request": request, "context": context},
                user_id=user_id,
            )

            if not authorized:
                return ExecutionResult(
                    request_id=request_id,
                    success=False,
                    result=None,
                    error=f"Authorization denied: {reason}",
                )

            # Analyze and plan
            plan = await self._analyze_and_plan(request, context)

            # Execute plan
            result = await self._execute_plan(plan, exec_context)

            # Store in memory
            if self._memory_system:
                await self._memory_system.store(
                    content=request,
                    metadata={
                        "type": "request",
                        "result": str(result)[:500],
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                )

            execution_time_ms = (time.monotonic() - start_time) * 1000

            # Audit
            await self._security.audit(
                operation="process_request",
                risk_level=RiskLevel.LOW,
                result="success",
                details={
                    "request": request[:100],
                    "plan_steps": len(plan) if plan else 0,
                },
                user_id=user_id,
                execution_time_ms=execution_time_ms,
            )

            return ExecutionResult(
                request_id=request_id,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                steps_executed=len(plan) if plan else 0,
            )

        except Exception as e:
            logger.error(
                "Request processing failed",
                request_id=request_id,
                error=str(e),
            )

            await self._security.audit(
                operation="process_request",
                risk_level=RiskLevel.LOW,
                result="failure",
                details={"request": request[:100]},
                user_id=user_id,
                error_message=str(e),
            )

            return ExecutionResult(
                request_id=request_id,
                success=False,
                result=None,
                error=str(e),
            )

        finally:
            async with self._request_lock:
                self._active_requests.pop(request_id, None)

    async def _analyze_and_plan(
        self,
        request: str,
        context: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Analyze request and create execution plan."""
        if not self._llm or not self._planning_graph:
            # Fallback to simple response
            return [{"type": "respond", "content": request}]

        # Use LLM to understand the request
        messages = [
            Message(
                role="system",
                content="""You are AION's planning module. Analyze the user request and create an execution plan.

Available actions:
- search_memory: Search for relevant information
- use_tool: Execute a specific tool
- reason: Perform reasoning/analysis
- respond: Generate a response
- vision: Process visual input

Output a JSON array of steps, each with:
- type: action type
- content: action details
- dependencies: list of step indices this depends on""",
            ),
            Message(role="user", content=request),
        ]

        try:
            response = await self._llm.complete(messages)

            # Parse plan from response
            import json
            try:
                plan = json.loads(response.content)
            except json.JSONDecodeError:
                plan = [{"type": "respond", "content": response.content}]

            return plan

        except Exception as e:
            logger.warning("Planning failed, using fallback", error=str(e))
            return [{"type": "respond", "content": request}]

    async def _execute_plan(
        self,
        plan: list[dict[str, Any]],
        context: ExecutionContext,
    ) -> Any:
        """Execute a plan step by step."""
        results = []

        for step in plan:
            step_type = step.get("type", "respond")

            if step_type == "search_memory" and self._memory_system:
                result = await self._memory_system.search(
                    query=step.get("content", ""),
                    limit=5,
                )
                results.append(result)

            elif step_type == "use_tool" and self._tool_orchestrator:
                result = await self._tool_orchestrator.execute(
                    tool_name=step.get("tool", ""),
                    params=step.get("params", {}),
                )
                results.append(result)

            elif step_type == "vision" and self._visual_cortex:
                result = await self._visual_cortex.process(
                    image_path=step.get("image", ""),
                    query=step.get("content", ""),
                )
                results.append(result)

            elif step_type == "reason" and self._llm:
                messages = [Message(role="user", content=step.get("content", ""))]
                response = await self._llm.complete(messages)
                results.append(response.content)

            else:
                results.append(step.get("content", ""))

        # Combine results
        if len(results) == 1:
            return results[0]
        return results

    # ==================== System Information ====================

    def get_status(self) -> dict[str, Any]:
        """Get current system status."""
        return {
            "instance_id": self.instance_id,
            "status": self._status.value,
            "uptime_seconds": (
                (datetime.now() - self._start_time).total_seconds()
                if self._start_time
                else 0
            ),
            "request_count": self._request_count,
            "active_requests": len(self._active_requests),
            "health": {
                name: {
                    "status": h.status.value,
                    "message": h.message,
                    "last_check": h.last_check.isoformat(),
                }
                for name, h in self._health.items()
            },
        }

    def get_health(self) -> dict[str, SystemHealth]:
        """Get health status of all subsystems."""
        return self._health.copy()

    def is_ready(self) -> bool:
        """Check if the kernel is ready to process requests."""
        return self._status == SystemStatus.READY

    # ==================== Subsystem Access ====================

    @property
    def security(self) -> SecurityManager:
        """Get the security manager."""
        if not self._security:
            raise RuntimeError("Kernel not initialized")
        return self._security

    @property
    def llm(self) -> LLMAdapter:
        """Get the LLM adapter."""
        if not self._llm:
            raise RuntimeError("Kernel not initialized")
        return self._llm

    @property
    def planning(self):
        """Get the planning graph."""
        return self._planning_graph

    @property
    def memory(self):
        """Get the memory system."""
        return self._memory_system

    @property
    def tools(self):
        """Get the tool orchestrator."""
        return self._tool_orchestrator

    @property
    def evolution(self):
        """Get the evolution engine."""
        return self._evolution_engine

    @property
    def vision(self):
        """Get the visual cortex."""
        return self._visual_cortex

    # ==================== Process Manager Access ====================

    @property
    def event_bus(self):
        """Get the event bus for pub/sub communication."""
        return self._event_bus

    @property
    def supervisor(self):
        """Get the process supervisor for agent management."""
        return self._supervisor

    @property
    def scheduler(self):
        """Get the task scheduler."""
        return self._scheduler

    @property
    def worker_pool(self):
        """Get the worker pool for background tasks."""
        return self._worker_pool

    @property
    def resource_manager(self):
        """Get the resource manager."""
        return self._resource_manager

    async def spawn_agent(self, config: "AgentConfig") -> str:
        """
        Convenience method to spawn an agent.

        Args:
            config: Agent configuration

        Returns:
            Process ID of the spawned agent
        """
        if not self._supervisor:
            raise RuntimeError("Process supervisor not initialized")
        return await self._supervisor.spawn_agent(config)

    async def schedule_task(
        self,
        name: str,
        handler: str,
        cron_expression: str = None,
        interval_seconds: int = None,
        run_at: datetime = None,
        params: dict = None,
    ) -> str:
        """
        Convenience method to schedule a task.

        Args:
            name: Task name
            handler: Handler function name
            cron_expression: Cron expression for scheduling
            interval_seconds: Interval for recurring tasks
            run_at: Datetime for one-time tasks
            params: Parameters to pass to handler

        Returns:
            Task ID
        """
        if not self._scheduler:
            raise RuntimeError("Task scheduler not initialized")

        if cron_expression:
            return await self._scheduler.schedule_cron(
                name=name,
                handler=handler,
                cron_expression=cron_expression,
                params=params,
            )
        elif interval_seconds:
            return await self._scheduler.schedule_interval(
                name=name,
                handler=handler,
                interval_seconds=interval_seconds,
                params=params,
            )
        elif run_at:
            return await self._scheduler.schedule_once(
                name=name,
                handler=handler,
                run_at=run_at,
                params=params,
            )
        else:
            raise ValueError("Must specify cron_expression, interval_seconds, or run_at")

    def get_process_stats(self) -> dict[str, Any]:
        """Get comprehensive process manager statistics."""
        stats = {}

        if self._supervisor:
            stats["supervisor"] = self._supervisor.get_stats()
        if self._scheduler:
            stats["scheduler"] = self._scheduler.get_stats()
        if self._worker_pool:
            stats["worker_pool"] = self._worker_pool.get_stats()
        if self._event_bus:
            stats["event_bus"] = self._event_bus.get_stats()
        if self._resource_manager:
            stats["resources"] = self._resource_manager.get_stats()

        return stats
