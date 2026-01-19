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

        # Shutdown subsystems
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
