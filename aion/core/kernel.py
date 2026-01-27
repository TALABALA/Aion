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

# MCP integration imports (conditional to avoid import errors)
try:
    from aion.mcp import MCPManager, ServerRegistry, CredentialManager, MCPServer
    from aion.mcp.server.aion_tools import setup_aion_mcp_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Persistence layer imports (conditional to avoid import errors)
try:
    from aion.persistence import (
        StateManager,
        PersistenceConfig,
        DatabaseBackend,
        BackupManager,
        TransactionManager,
        MigrationRunner,
    )
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

# Goal system imports (conditional to avoid import errors)
try:
    from aion.systems.goals import AutonomousGoalManager
    GOAL_SYSTEM_AVAILABLE = True
except ImportError:
    GOAL_SYSTEM_AVAILABLE = False

# Automation system imports (conditional to avoid import errors)
try:
    from aion.automation.engine import WorkflowEngine
    from aion.automation.triggers.manager import TriggerManager
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False

# Plugin system imports (conditional to avoid import errors)
try:
    from aion.plugins import PluginManager, PluginSystemConfig
    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False

# Distributed computing imports (conditional to avoid import errors)
try:
    from aion.distributed.cluster.manager import ClusterManager
    from aion.distributed.config import DistributedConfig
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
# Reinforcement Learning Loop imports (conditional to avoid import errors)
try:
    from aion.learning import ReinforcementLearningLoop, LearningConfig
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

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
        self._audio_cortex = None

        # Process Manager components
        self._event_bus = None
        self._supervisor = None
        self._scheduler = None
        self._worker_pool = None
        self._resource_manager = None

        # MCP Integration components
        self._mcp_manager = None
        self._mcp_server = None

        # Conversation system
        self._conversation = None

        # Goal system
        self._goal_manager = None

        # Automation system
        self._automation_engine = None
        self._trigger_manager = None

        # Plugin system
        self._plugin_manager = None

        # Distributed computing
        self._cluster_manager = None
        # Reinforcement Learning Loop
        self._rl_loop = None

        # Persistence layer
        self._state_manager = None
        self._backup_manager = None
        self._transaction_manager = None

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

                # Initialize persistence layer (before subsystems)
                if PERSISTENCE_AVAILABLE:
                    await self._initialize_persistence()

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

    async def _initialize_persistence(self) -> None:
        """Initialize the persistence layer."""
        try:
            # Create persistence config
            persistence_config = PersistenceConfig(
                backend=DatabaseBackend.SQLITE,
            )
            persistence_config.sqlite.path = self.config.data_dir / "aion.db"

            # Initialize state manager
            self._state_manager = StateManager(persistence_config)
            await self._state_manager.initialize()
            self._update_health("persistence", SystemStatus.READY)

            # Initialize backup manager
            self._backup_manager = BackupManager(
                connection=self._state_manager._db,
                backup_dir=self.config.data_dir / "backups",
            )
            await self._backup_manager.initialize()
            self._update_health("backup", SystemStatus.READY)

            # Initialize transaction manager
            self._transaction_manager = TransactionManager(
                connection=self._state_manager._db,
            )
            await self._transaction_manager.initialize()

            # Run migrations
            migration_runner = MigrationRunner(connection=self._state_manager._db)
            await migration_runner.initialize()
            migration_result = await migration_runner.migrate()
            if migration_result.applied:
                logger.info(
                    "Applied database migrations",
                    count=len(migration_result.applied),
                )

            logger.info(
                "Persistence layer initialized successfully",
                backend=persistence_config.backend.value,
            )

        except Exception as e:
            logger.error("Persistence layer initialization failed", error=str(e))
            self._update_health("persistence", SystemStatus.ERROR, str(e))

    async def _initialize_subsystems(self) -> None:
        """Initialize all cognitive subsystems."""
        # Import here to avoid circular imports
        from aion.systems.planning import PlanningGraph
        from aion.systems.memory import CognitiveMemorySystem
        from aion.systems.tools import ToolOrchestrator
        from aion.systems.evolution import SelfImprovementEngine
        from aion.systems.vision import VisualCortex
        from aion.systems.audio import AuditoryCortex, AuditoryCortexConfig

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

        # Initialize MCP Integration
        if MCP_AVAILABLE and self.config.mcp.enabled:
            await self._initialize_mcp()

        # Initialize Conversation System
        await self._initialize_conversation()

        # Initialize Goal System
        if GOAL_SYSTEM_AVAILABLE:
            await self._initialize_goal_system()

        # Initialize Automation System
        if AUTOMATION_AVAILABLE:
            await self._initialize_automation()

        # Initialize Plugin System
        if PLUGINS_AVAILABLE:
            await self._initialize_plugins()

        # Initialize Distributed Computing System
        if DISTRIBUTED_AVAILABLE and self.config.distributed_enabled:
            await self._initialize_distributed()
        # Initialize Reinforcement Learning Loop
        if LEARNING_AVAILABLE:
            await self._initialize_learning()

    async def _initialize_conversation(self) -> None:
        """Initialize the Conversation System."""
        try:
            from aion.conversation import (
                ConversationManager,
                ConversationConfig,
            )
            from aion.conversation.memory.integrator import MemoryIntegrator
            from aion.conversation.tools.executor import ToolExecutor
            from aion.conversation.llm.claude import ClaudeProvider

            # Create memory integrator if memory system is available
            memory_integrator = None
            if self._memory_system:
                memory_integrator = MemoryIntegrator(self._memory_system)

            # Create tool executor if tool orchestrator is available
            tool_executor = None
            if self._tool_orchestrator:
                tool_executor = ToolExecutor(self._tool_orchestrator)

            # Create LLM provider
            llm_provider = ClaudeProvider(
                api_key=self.config.get_llm_api_key(),
                default_model=self.config.llm.model,
            )

            # Create default conversation config
            default_config = ConversationConfig(
                model=self.config.llm.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
            )

            # Create conversation manager
            self._conversation = ConversationManager(
                llm_provider=llm_provider,
                memory_integrator=memory_integrator,
                tool_executor=tool_executor,
                default_config=default_config,
            )

            await self._conversation.initialize()
            self._update_health("conversation", SystemStatus.READY)
            logger.info("Conversation system initialized successfully")

        except ImportError as e:
            logger.warning(f"Conversation system not available: {e}")
            self._update_health("conversation", SystemStatus.DEGRADED, "Not available")
        except Exception as e:
            logger.error(f"Conversation system initialization failed: {e}")
            self._update_health("conversation", SystemStatus.ERROR, str(e))

    async def _initialize_goal_system(self) -> None:
        """Initialize the Autonomous Goal System."""
        try:
            from aion.systems.goals import AutonomousGoalManager
            from aion.systems.goals.executor import GoalExecutor

            # Create executor with available subsystems
            executor = GoalExecutor(
                planning_engine=self._planning_graph,
                process_supervisor=self._supervisor,
                tool_orchestrator=self._tool_orchestrator,
                llm_provider=self._conversation.llm if self._conversation else None,
            )

            # Create goal manager
            self._goal_manager = AutonomousGoalManager(
                executor=executor,
                auto_generate_goals=self.config.evolution.enable_self_improvement,
                data_dir=str(self.config.data_dir / "goals"),
            )

            await self._goal_manager.initialize()
            self._update_health("goals", SystemStatus.READY)
            logger.info("Goal system initialized successfully")

        except ImportError as e:
            logger.warning(f"Goal system not available: {e}")
            self._update_health("goals", SystemStatus.DEGRADED, "Not available")
        except Exception as e:
            logger.error(f"Goal system initialization failed: {e}")
            self._update_health("goals", SystemStatus.ERROR, str(e))

    async def _initialize_automation(self) -> None:
        """Initialize the Workflow Automation System."""
        try:
            from aion.automation.engine import WorkflowEngine
            from aion.automation.triggers.manager import TriggerManager

            # Create trigger manager
            self._trigger_manager = TriggerManager()

            # Create workflow engine
            self._automation_engine = WorkflowEngine(
                trigger_manager=self._trigger_manager,
            )

            # Connect to event bus if available
            if self._event_bus:
                self._automation_engine.set_event_bus(self._event_bus)

            await self._automation_engine.initialize()
            self._update_health("automation", SystemStatus.READY)
            logger.info("Automation system initialized successfully")

        except ImportError as e:
            logger.warning(f"Automation system not available: {e}")
            self._update_health("automation", SystemStatus.DEGRADED, "Not available")
        except Exception as e:
            logger.error(f"Automation system initialization failed: {e}")
            self._update_health("automation", SystemStatus.ERROR, str(e))

    async def _initialize_plugins(self) -> None:
        """Initialize the Plugin System."""
        try:
            from aion.plugins import PluginManager, PluginSystemConfig

            # Create plugin system config
            plugin_config = PluginSystemConfig(
                plugins_dir=self.config.data_dir / "plugins",
                enabled=True,
                auto_discover=True,
                hot_reload=False,  # Disabled by default for stability
                sandbox_enabled=True,
            )

            # Create plugin manager
            self._plugin_manager = PluginManager(
                kernel=self,
                config=plugin_config,
            )

            await self._plugin_manager.initialize()
            self._update_health("plugins", SystemStatus.READY)

            # Bridge plugin tools to tool orchestrator if available
            if self._tool_orchestrator:
                plugin_tools = self._plugin_manager.get_plugin_tools()
                for tool in plugin_tools:
                    try:
                        await self._tool_orchestrator.register_tool(
                            name=tool.name,
                            handler=tool.handler,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to register plugin tool: {tool.name}",
                            error=str(e),
                        )

            logger.info(
                "Plugin system initialized successfully",
                plugins_loaded=len(self._plugin_manager.list_plugins()),
            )

        except ImportError as e:
            logger.warning(f"Plugin system not available: {e}")
            self._update_health("plugins", SystemStatus.DEGRADED, "Not available")
        except Exception as e:
            logger.error(f"Plugin system initialization failed: {e}")
            self._update_health("plugins", SystemStatus.ERROR, str(e))

    async def _initialize_distributed(self) -> None:
        """Initialize the Distributed Computing System."""
        try:
            from aion.distributed.cluster.manager import ClusterManager
            from aion.distributed.config import DistributedConfig

            # Create distributed config
            dist_config = DistributedConfig(
                node_name=self.instance_id,
            )

            # Create cluster manager
            self._cluster_manager = ClusterManager(
                kernel=self,
                config=dist_config.to_flat_dict(),
            )

            await self._cluster_manager.start()
            self._update_health("distributed", SystemStatus.READY)
            logger.info(
                "Distributed computing system initialized",
                node_id=self._cluster_manager.node_info.id,
                node_name=self._cluster_manager.node_info.name,
            )

        except ImportError as e:
            logger.warning(f"Distributed system not available: {e}")
            self._update_health("distributed", SystemStatus.DEGRADED, "Not available")
        except Exception as e:
            logger.error(f"Distributed system initialization failed: {e}")
            self._update_health("distributed", SystemStatus.ERROR, str(e))
    async def _initialize_learning(self) -> None:
        """Initialize the Reinforcement Learning Loop."""
        try:
            from aion.learning import ReinforcementLearningLoop, LearningConfig

            self._rl_loop = ReinforcementLearningLoop(
                kernel=self,
                config=LearningConfig(),
            )
            await self._rl_loop.initialize()
            self._update_health("learning", SystemStatus.READY)
            logger.info("Reinforcement Learning Loop initialized successfully")

        except ImportError as e:
            logger.warning(f"Learning system not available: {e}")
            self._update_health("learning", SystemStatus.DEGRADED, "Not available")
        except Exception as e:
            logger.error(f"Learning system initialization failed: {e}")
            self._update_health("learning", SystemStatus.ERROR, str(e))

    async def _initialize_mcp(self) -> None:
        """Initialize the MCP Integration Layer."""
        mcp_config = self.config.mcp

        try:
            # Create registry and credentials manager
            registry = ServerRegistry(mcp_config.config_path)
            credentials = CredentialManager(mcp_config.credentials_path)

            # Create and initialize MCP manager
            self._mcp_manager = MCPManager(
                registry=registry,
                credentials=credentials,
                health_check_interval=mcp_config.health_check_interval,
                auto_reconnect=mcp_config.auto_reconnect,
                max_reconnect_attempts=mcp_config.max_reconnect_attempts,
            )
            await self._mcp_manager.initialize()
            self._update_health("mcp", SystemStatus.READY)

            # Bridge MCP tools to AION's tool system
            if self._tool_orchestrator:
                bridge = self._mcp_manager.get_tool_bridge()
                bridge.register_with_orchestrator(self._tool_orchestrator)

            # Optionally serve as MCP server
            if mcp_config.serve_as_mcp_server:
                self._mcp_server = MCPServer(
                    kernel=self,
                    server_name=mcp_config.mcp_server_name,
                    server_version=mcp_config.mcp_server_version,
                )
                setup_aion_mcp_server(self._mcp_server, self)
                self._update_health("mcp_server", SystemStatus.READY)

            logger.info(
                "MCP Integration initialized successfully",
                servers_connected=len(self._mcp_manager.get_connected_servers()),
            )

        except Exception as e:
            logger.error("MCP Integration initialization failed", error=str(e))
            self._update_health("mcp", SystemStatus.ERROR, str(e))

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
        # Initialize Auditory Cortex
        if self.config.audio.enabled:
            try:
                audio_config = AuditoryCortexConfig(
                    whisper_model=self.config.audio.whisper_model,
                    event_detection_model=self.config.audio.event_detection_model,
                    speaker_embedding_model=self.config.audio.speaker_embedding_model,
                    clap_model=self.config.audio.clap_model,
                    tts_model=self.config.audio.tts_model,
                    enable_tts=self.config.audio.enable_tts,
                    enable_diarization=self.config.audio.enable_diarization,
                    diarization_model=self.config.audio.diarization_model,
                    enable_memory=self.config.audio.enable_memory,
                    memory_embedding_dim=self.config.audio.memory_embedding_dim,
                    memory_max_entries=self.config.audio.memory_max_entries,
                    memory_index_path=self.config.audio.memory_index_path,
                    device=self.config.audio.device,
                    target_sample_rate=self.config.audio.target_sample_rate,
                    max_audio_duration=self.config.audio.max_audio_duration,
                    enable_music_analysis=self.config.audio.enable_music_analysis,
                    event_threshold=self.config.audio.event_threshold,
                )
                self._audio_cortex = AuditoryCortex(config=audio_config)
                await self._audio_cortex.initialize()
                self._update_health("audio", SystemStatus.READY)
            except Exception as e:
                logger.warning("Auditory cortex initialization failed", error=str(e))
                self._update_health("audio", SystemStatus.ERROR, str(e))

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

        # Shutdown MCP integration
        if self._mcp_manager:
            await self._mcp_manager.shutdown()

        # Shutdown conversation system
        if self._conversation:
            await self._conversation.shutdown()

        # Shutdown goal system
        if self._goal_manager:
            await self._goal_manager.shutdown()

        # Shutdown automation system
        if self._automation_engine:
            await self._automation_engine.shutdown()
        if self._trigger_manager:
            await self._trigger_manager.shutdown()

        # Shutdown plugin system
        if self._plugin_manager:
            await self._plugin_manager.shutdown()

        # Shutdown distributed computing
        if self._cluster_manager:
            await self._cluster_manager.stop()
        # Shutdown learning loop
        if self._rl_loop:
            await self._rl_loop.shutdown()

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
        if self._audio_cortex:
            await self._audio_cortex.shutdown()

        # Shutdown persistence layer (last)
        if self._state_manager:
            # Create backup on shutdown if configured
            if self._backup_manager:
                try:
                    await self._backup_manager.create_backup(name="shutdown_backup")
                    logger.info("Shutdown backup created")
                except Exception as e:
                    logger.warning("Failed to create shutdown backup", error=str(e))

            await self._state_manager.shutdown()

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

            elif step_type == "audio" and self._audio_cortex:
                result = await self._audio_cortex.understand_scene(
                    audio=step.get("audio", ""),
                )
                results.append(result)

            elif step_type == "transcribe" and self._audio_cortex:
                result = await self._audio_cortex.transcribe(
                    audio=step.get("audio", ""),
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
    @property
    def audio(self):
        """Get the auditory cortex."""
        return self._audio_cortex

    @property
    def conversation(self):
        """Get the conversation manager."""
        return self._conversation

    def get_conversation_stats(self) -> dict[str, Any]:
        """Get conversation system statistics."""
        if not self._conversation:
            return {"available": False}

        return {
            "available": True,
            **self._conversation.get_stats(),
        }

    # ==================== MCP Integration Access ====================

    @property
    def mcp(self):
        """Get the MCP manager for external tool integration."""
        return self._mcp_manager

    @property
    def mcp_server(self):
        """Get the MCP server (when AION is serving as MCP server)."""
        return self._mcp_server

    def get_mcp_stats(self) -> dict[str, Any]:
        """Get MCP integration statistics."""
        if not self._mcp_manager:
            return {"available": False}

        return {
            "available": True,
            **self._mcp_manager.get_stats(),
        }

    # ==================== Goal System Access ====================

    @property
    def goals(self):
        """Get the autonomous goal manager."""
        return self._goal_manager

    def get_goal_stats(self) -> dict[str, Any]:
        """Get goal system statistics."""
        if not self._goal_manager:
            return {"available": False}

        return {
            "available": True,
            **self._goal_manager.get_stats(),
        }

    # ==================== Automation System Access ====================

    @property
    def automation(self):
        """Get the workflow automation engine."""
        return self._automation_engine

    @property
    def triggers(self):
        """Get the trigger manager."""
        return self._trigger_manager

    def get_automation_stats(self) -> dict[str, Any]:
        """Get automation system statistics."""
        if not self._automation_engine:
            return {"available": False}

        return {
            "available": True,
            **self._automation_engine.get_stats(),
        }

    # ==================== Plugin System Access ====================

    @property
    def plugins(self):
        """Get the plugin manager."""
        return self._plugin_manager

    async def load_plugin(
        self,
        plugin_id: str,
        config: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Convenience method to load and activate a plugin.

        Args:
            plugin_id: Plugin identifier
            config: Optional plugin configuration

        Returns:
            True if successful
        """
        if not self._plugin_manager:
            return False

        try:
            await self._plugin_manager.load(plugin_id, config=config or {})
            await self._plugin_manager.activate(plugin_id)
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin: {plugin_id}", error=str(e))
            return False

    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        Convenience method to unload a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if successful
        """
        if not self._plugin_manager:
            return False

        try:
            await self._plugin_manager.unload(plugin_id)
            return True
        except Exception as e:
            logger.error(f"Failed to unload plugin: {plugin_id}", error=str(e))
            return False

    def get_plugin_stats(self) -> dict[str, Any]:
        """Get plugin system statistics."""
        if not self._plugin_manager:
            return {"available": False}

        return {
            "available": True,
            **self._plugin_manager.get_stats(),
        }

    def get_plugin_tools(self) -> list:
        """Get all tools registered by active plugins."""
        if not self._plugin_manager:
            return []
        return self._plugin_manager.get_plugin_tools()

    # ==================== Distributed Computing Access ====================

    @property
    def cluster(self):
        """Get the cluster manager for distributed operations."""
        return self._cluster_manager

    def get_cluster_stats(self) -> dict[str, Any]:
        """Get distributed cluster statistics."""
        if not self._cluster_manager:
            return {"available": False}
        return self._cluster_manager.get_stats()

    # ==================== Reinforcement Learning Access ====================

    @property
    def learning(self):
        """Get the reinforcement learning loop."""
        return self._rl_loop

    def get_learning_stats(self) -> dict[str, Any]:
        """Get reinforcement learning loop statistics."""
        if not self._rl_loop:
            return {"available": False}

        return {
            "available": True,
            **self._cluster_manager.get_stats(),
        }

    def get_cluster_info(self) -> dict[str, Any]:
        """Get cluster information."""
        if not self._cluster_manager:
            return {"available": False}

        return {
            "available": True,
            **self._cluster_manager.get_cluster_info(),
            **self._rl_loop.get_stats(),
        }

    # ==================== Persistence Layer Access ====================

    @property
    def state(self):
        """Get the state manager for persistence operations."""
        return self._state_manager

    @property
    def backup(self):
        """Get the backup manager."""
        return self._backup_manager

    @property
    def transactions(self):
        """Get the transaction manager."""
        return self._transaction_manager

    async def create_backup(self, name: Optional[str] = None) -> Optional[str]:
        """
        Convenience method to create a backup.

        Args:
            name: Optional backup name

        Returns:
            Backup ID if successful, None otherwise
        """
        if not self._backup_manager:
            return None

        try:
            metadata = await self._backup_manager.create_backup(name=name)
            return metadata.id
        except Exception as e:
            logger.error("Backup creation failed", error=str(e))
            return None

    async def restore_backup(self, backup_id: str) -> bool:
        """
        Convenience method to restore from a backup.

        Args:
            backup_id: ID of backup to restore

        Returns:
            True if successful
        """
        if not self._backup_manager:
            return False

        try:
            result = await self._backup_manager.restore(backup_id)
            return result.success
        except Exception as e:
            logger.error("Backup restoration failed", error=str(e))
            return False

    def get_persistence_stats(self) -> dict[str, Any]:
        """Get persistence layer statistics."""
        if not self._state_manager:
            return {"available": False}

        stats = {
            "available": True,
            "initialized": self._state_manager._initialized,
        }

        if self._backup_manager:
            try:
                # Get backup stats synchronously if possible
                stats["backups"] = {"available": True}
            except Exception:
                stats["backups"] = {"available": False}

        return stats
