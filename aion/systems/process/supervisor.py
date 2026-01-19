"""
AION Process Supervisor

State-of-the-art process lifecycle management with:
- Full lifecycle control (spawn, start, pause, resume, stop, terminate)
- Health monitoring and automatic restart policies
- Resource limit enforcement with graceful degradation
- Process hierarchy management (parent/child relationships)
- Event-driven state transitions
- Graceful and forced shutdown support
- Comprehensive metrics and statistics
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
import psutil
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Type, Dict, List, Set
from weakref import WeakValueDictionary
from collections import defaultdict

import structlog

from aion.systems.process.models import (
    ProcessInfo,
    ProcessState,
    ProcessType,
    ProcessPriority,
    RestartPolicy,
    ResourceLimits,
    ResourceUsage,
    AgentConfig,
    Event,
    ProcessCheckpoint,
    SupervisorStats,
    SignalType,
)
from aion.systems.process.agent_base import BaseAgent
from aion.systems.process.event_bus import EventBus

logger = structlog.get_logger(__name__)


class ProcessSupervisor:
    """
    Central process supervisor for AION.

    Responsibilities:
    - Spawn and manage process lifecycles
    - Monitor health and enforce resource limits
    - Handle failures and implement restart policies
    - Maintain process hierarchy (parent/child)
    - Coordinate graceful and forced shutdown
    - Emit lifecycle events for observability
    """

    def __init__(
        self,
        event_bus: EventBus,
        kernel: Optional[Any] = None,
        health_check_interval: float = 5.0,
        max_processes: int = 100,
        default_restart_delay: float = 1.0,
        default_max_restarts: int = 5,
        enable_resource_monitoring: bool = True,
        zombie_timeout_seconds: float = 300.0,
    ):
        self.event_bus = event_bus
        self.kernel = kernel
        self.health_check_interval = health_check_interval
        self.max_processes = max_processes
        self.default_restart_delay = default_restart_delay
        self.default_max_restarts = default_max_restarts
        self.enable_resource_monitoring = enable_resource_monitoring
        self.zombie_timeout_seconds = zombie_timeout_seconds

        # Process tracking
        self._processes: Dict[str, ProcessInfo] = {}
        self._process_tasks: Dict[str, asyncio.Task] = {}
        self._process_instances: WeakValueDictionary[str, BaseAgent] = WeakValueDictionary()

        # Agent class registry
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}

        # Process groups for batch operations
        self._process_groups: Dict[str, Set[str]] = defaultdict(set)

        # Lifecycle hooks
        self._on_process_start: List[Callable] = []
        self._on_process_stop: List[Callable] = []
        self._on_process_fail: List[Callable] = []
        self._on_state_change: List[Callable] = []

        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Checkpoint storage
        self._checkpoints: Dict[str, List[ProcessCheckpoint]] = defaultdict(list)
        self._max_checkpoints_per_process: int = 10

        # Statistics
        self._stats = SupervisorStats()
        self._start_time: Optional[datetime] = None

        # Locks
        self._spawn_lock = asyncio.Lock()
        self._process_locks: Dict[str, asyncio.Lock] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the supervisor."""
        if self._initialized:
            return

        logger.info("Initializing Process Supervisor")
        self._start_time = datetime.now()

        # Start background tasks
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Register built-in agent classes (deferred import to avoid circular deps)
        self._register_builtin_agents()

        # Subscribe to relevant events
        await self.event_bus.subscribe(
            "process.*",
            self._on_process_event,
            subscriber_id="supervisor",
        )

        self._initialized = True
        logger.info(
            "Process Supervisor initialized",
            max_processes=self.max_processes,
            health_check_interval=self.health_check_interval,
        )

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shutdown all processes."""
        logger.info("Shutting down Process Supervisor")

        self._shutdown_event.set()

        # Stop background tasks
        for task in [self._health_monitor_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

        # Stop all processes by priority (lowest priority first)
        processes_by_priority = sorted(
            [p for p in self._processes.values() if p.state.is_active()],
            key=lambda p: -p.priority.value,  # Higher value = lower priority = stop first
        )

        # Calculate per-process timeout
        per_process_timeout = timeout / max(len(processes_by_priority), 1)

        for process_info in processes_by_priority:
            try:
                await self.stop_process(
                    process_info.id,
                    graceful=True,
                    timeout=per_process_timeout,
                )
            except Exception as e:
                logger.error(
                    "Failed to stop process during shutdown",
                    process_id=process_info.id,
                    error=str(e),
                )

        # Force terminate any remaining
        for process_id, task in list(self._process_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

        logger.info(
            "Process Supervisor shutdown complete",
            processes_stopped=self._stats.processes_terminated,
        )

    def register_agent_class(
        self,
        name: str,
        agent_class: Type[BaseAgent],
    ) -> None:
        """Register an agent class for spawning."""
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"{agent_class} must be a subclass of BaseAgent")
        self._agent_classes[name] = agent_class
        logger.debug(f"Registered agent class: {name}")

    def get_registered_classes(self) -> List[str]:
        """Get list of registered agent class names."""
        return list(self._agent_classes.keys())

    async def spawn_agent(
        self,
        config: AgentConfig,
        parent_id: Optional[str] = None,
        group: Optional[str] = None,
    ) -> str:
        """
        Spawn a new agent process.

        Args:
            config: Agent configuration
            parent_id: Parent process ID (for hierarchy)
            group: Optional group name for batch operations

        Returns:
            Process ID of the spawned agent

        Raises:
            RuntimeError: If max processes reached or parent can't spawn
            ValueError: If agent class not found
        """
        async with self._spawn_lock:
            # Check global limits
            active_count = len([p for p in self._processes.values() if p.state.is_active()])
            if active_count >= self.max_processes:
                raise RuntimeError(
                    f"Maximum process limit reached: {active_count}/{self.max_processes}"
                )

            # Check parent limits
            if parent_id:
                parent = self._processes.get(parent_id)
                if parent:
                    active_children = len([
                        c for c in parent.children_ids
                        if c in self._processes and self._processes[c].state.is_active()
                    ])
                    if active_children >= parent.limits.max_child_processes:
                        raise RuntimeError(
                            f"Parent {parent_id} has reached child limit: "
                            f"{active_children}/{parent.limits.max_child_processes}"
                        )

            # Get agent class
            agent_class = self._agent_classes.get(config.agent_class)
            if not agent_class:
                raise ValueError(f"Unknown agent class: {config.agent_class}")

            # Create process info
            process_id = str(uuid.uuid4())
            now = datetime.now()

            process_info = ProcessInfo(
                id=process_id,
                name=config.name,
                type=ProcessType.AGENT,
                state=ProcessState.CREATED,
                priority=config.priority,
                created_at=now,
                parent_id=parent_id,
                restart_policy=config.restart_policy,
                max_restarts=config.max_restarts,
                restart_delay_seconds=config.restart_delay_seconds,
                limits=config.limits,
                agent_class=config.agent_class,
                agent_config_hash=config.get_config_hash(),
                input_channels=config.input_channels.copy(),
                output_channels=config.output_channels.copy(),
                metadata=config.metadata.copy(),
                tags=config.tags.copy(),
            )

            self._processes[process_id] = process_info
            self._process_locks[process_id] = asyncio.Lock()

            # Update parent's children list
            if parent_id and parent_id in self._processes:
                self._processes[parent_id].children_ids.append(process_id)

            # Add to group if specified
            if group:
                self._process_groups[group].add(process_id)

        # Create agent instance
        agent = agent_class(
            process_id=process_id,
            config=config,
            supervisor=self,
            event_bus=self.event_bus,
            kernel=self.kernel,
        )
        self._process_instances[process_id] = agent

        # Start agent
        await self._start_process(process_id, agent)

        self._stats.processes_spawned += 1
        self._update_stats()

        logger.info(
            "Spawned agent",
            process_id=process_id,
            name=config.name,
            agent_class=config.agent_class,
            parent_id=parent_id,
        )

        return process_id

    async def spawn_task(
        self,
        name: str,
        handler: Callable,
        params: Optional[dict] = None,
        priority: ProcessPriority = ProcessPriority.NORMAL,
        limits: Optional[ResourceLimits] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Spawn a one-shot task process.

        Args:
            name: Task name
            handler: Async function to execute
            params: Parameters to pass to handler
            priority: Task priority
            limits: Resource limits
            timeout: Task timeout in seconds

        Returns:
            Process ID
        """
        process_id = str(uuid.uuid4())
        now = datetime.now()

        process_info = ProcessInfo(
            id=process_id,
            name=name,
            type=ProcessType.TASK,
            state=ProcessState.CREATED,
            priority=priority,
            created_at=now,
            restart_policy=RestartPolicy.NEVER,
            limits=limits or ResourceLimits(),
        )

        self._processes[process_id] = process_info
        self._process_locks[process_id] = asyncio.Lock()

        # Create task wrapper
        async def task_wrapper():
            try:
                process_info.state = ProcessState.STARTING
                process_info.started_at = datetime.now()

                # Emit start event
                await self._emit_lifecycle_event("started", process_id)

                process_info.state = ProcessState.RUNNING

                # Execute with optional timeout
                if timeout:
                    result = await asyncio.wait_for(
                        handler(**(params or {})),
                        timeout=timeout,
                    )
                else:
                    result = await handler(**(params or {}))

                process_info.state = ProcessState.STOPPED
                process_info.stopped_at = datetime.now()
                process_info.exit_code = 0

                await self._emit_lifecycle_event("stopped", process_id)

                return result

            except asyncio.TimeoutError:
                process_info.state = ProcessState.FAILED
                process_info.stopped_at = datetime.now()
                process_info.error = f"Task timeout after {timeout}s"
                process_info.exit_code = -1

                await self._emit_lifecycle_event("failed", process_id, error=process_info.error)
                raise

            except asyncio.CancelledError:
                process_info.state = ProcessState.TERMINATED
                process_info.stopped_at = datetime.now()
                raise

            except Exception as e:
                process_info.state = ProcessState.FAILED
                process_info.stopped_at = datetime.now()
                process_info.error = str(e)
                process_info.error_traceback = traceback.format_exc()
                process_info.exit_code = 1

                await self._emit_lifecycle_event("failed", process_id, error=str(e))
                raise

        # Start task
        task = asyncio.create_task(task_wrapper())
        self._process_tasks[process_id] = task

        self._stats.processes_spawned += 1
        self._update_stats()

        return process_id

    async def _start_process(
        self,
        process_id: str,
        agent: BaseAgent,
    ) -> None:
        """Start an agent process."""
        process_info = self._processes[process_id]

        async with self._process_locks[process_id]:
            if not process_info.update_state(ProcessState.STARTING):
                raise RuntimeError(
                    f"Cannot start process in state {process_info.state.value}"
                )

            try:
                # Initialize agent
                await agent.initialize()

                # Create run task
                async def run_wrapper():
                    try:
                        process_info.state = ProcessState.RUNNING
                        process_info.started_at = datetime.now()

                        # Emit start event
                        await self._emit_lifecycle_event("started", process_id)

                        # Call lifecycle hooks
                        for hook in self._on_process_start:
                            try:
                                await self._call_hook(hook, process_info)
                            except Exception as e:
                                logger.warning(f"Start hook failed: {e}")

                        # Run agent
                        await agent.run()

                        # Normal completion
                        process_info.state = ProcessState.STOPPED
                        process_info.stopped_at = datetime.now()
                        process_info.exit_code = 0

                        await self._emit_lifecycle_event("stopped", process_id)

                    except asyncio.CancelledError:
                        process_info.state = ProcessState.TERMINATED
                        process_info.stopped_at = datetime.now()
                        await self._emit_lifecycle_event("terminated", process_id)
                        raise

                    except Exception as e:
                        process_info.state = ProcessState.FAILED
                        process_info.stopped_at = datetime.now()
                        process_info.error = str(e)
                        process_info.error_traceback = traceback.format_exc()
                        process_info.exit_code = 1

                        logger.error(
                            "Process failed",
                            process_id=process_id,
                            error=str(e),
                            traceback=process_info.error_traceback,
                        )

                        await self._emit_lifecycle_event("failed", process_id, error=str(e))

                        # Handle restart policy
                        await self._handle_process_failure(process_id)

                    finally:
                        # Call stop hooks
                        for hook in self._on_process_stop:
                            try:
                                await self._call_hook(hook, process_info)
                            except Exception as e:
                                logger.warning(f"Stop hook failed: {e}")

                task = asyncio.create_task(run_wrapper())
                self._process_tasks[process_id] = task

            except Exception as e:
                process_info.state = ProcessState.FAILED
                process_info.error = str(e)
                process_info.error_traceback = traceback.format_exc()
                raise

    async def stop_process(
        self,
        process_id: str,
        graceful: bool = True,
        timeout: float = 30.0,
    ) -> bool:
        """
        Stop a process.

        Args:
            process_id: Process to stop
            graceful: If True, allow graceful shutdown
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully
        """
        process_info = self._processes.get(process_id)
        if not process_info:
            return False

        if process_info.state.is_terminal():
            return True

        async with self._process_locks.get(process_id, asyncio.Lock()):
            if not process_info.state.is_active():
                return True

            # Mark as stopping
            process_info.state = ProcessState.STOPPING

            # Stop children first (with reduced timeout)
            child_timeout = timeout / (len(process_info.children_ids) + 1)
            for child_id in process_info.children_ids[:]:
                await self.stop_process(child_id, graceful, child_timeout)

            # Get agent instance
            agent = self._process_instances.get(process_id)
            task = self._process_tasks.get(process_id)

            try:
                if graceful and agent:
                    # Request graceful shutdown
                    try:
                        await asyncio.wait_for(agent.shutdown(), timeout=timeout * 0.8)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Graceful shutdown timed out",
                            process_id=process_id,
                            timeout=timeout,
                        )

                # Cancel task if still running
                if task and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=timeout * 0.2)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass

                process_info.state = ProcessState.STOPPED
                process_info.stopped_at = datetime.now()

                self._stats.processes_terminated += 1
                self._update_stats()

                logger.info(f"Stopped process: {process_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to stop process {process_id}: {e}")
                process_info.state = ProcessState.FAILED
                process_info.error = str(e)
                return False

    async def kill_process(self, process_id: str) -> bool:
        """Force terminate a process immediately."""
        process_info = self._processes.get(process_id)
        if not process_info:
            return False

        task = self._process_tasks.get(process_id)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        process_info.state = ProcessState.TERMINATED
        process_info.stopped_at = datetime.now()

        # Kill children
        for child_id in process_info.children_ids:
            await self.kill_process(child_id)

        self._stats.processes_terminated += 1
        self._update_stats()

        logger.info(f"Killed process: {process_id}")
        return True

    async def pause_process(self, process_id: str) -> bool:
        """Pause a running process."""
        process_info = self._processes.get(process_id)
        if not process_info or process_info.state != ProcessState.RUNNING:
            return False

        agent = self._process_instances.get(process_id)
        if agent:
            await agent.pause()

        process_info.state = ProcessState.PAUSED
        await self._emit_state_change(process_id, ProcessState.RUNNING, ProcessState.PAUSED)

        return True

    async def resume_process(self, process_id: str) -> bool:
        """Resume a paused process."""
        process_info = self._processes.get(process_id)
        if not process_info or process_info.state != ProcessState.PAUSED:
            return False

        agent = self._process_instances.get(process_id)
        if agent:
            await agent.resume()

        process_info.state = ProcessState.RUNNING
        await self._emit_state_change(process_id, ProcessState.PAUSED, ProcessState.RUNNING)

        return True

    async def restart_process(self, process_id: str) -> bool:
        """Restart a process."""
        process_info = self._processes.get(process_id)
        if not process_info:
            return False

        # Stop if running
        if process_info.state.is_active():
            await self.stop_process(process_id, graceful=True)

        # Reset state
        process_info.state = ProcessState.CREATED
        process_info.error = None
        process_info.error_traceback = None
        process_info.exit_code = None
        process_info.restart_count += 1
        process_info.last_restart_at = datetime.now()

        # Restart
        agent = self._process_instances.get(process_id)
        if agent:
            await self._start_process(process_id, agent)
            self._stats.restarts_performed += 1
            return True

        return False

    async def send_signal(
        self,
        process_id: str,
        signal: SignalType,
        payload: Optional[dict] = None,
    ) -> bool:
        """Send a signal to a process."""
        agent = self._process_instances.get(process_id)
        if not agent:
            return False

        await agent.receive_signal(signal, payload or {})
        self._stats.signals_sent += 1
        return True

    async def _handle_process_failure(self, process_id: str) -> None:
        """Handle a failed process according to restart policy."""
        process_info = self._processes.get(process_id)
        if not process_info:
            return

        self._stats.processes_failed += 1
        self._update_stats()

        # Call failure hooks
        for hook in self._on_process_fail:
            try:
                await self._call_hook(hook, process_info)
            except Exception as e:
                logger.warning(f"Failure hook failed: {e}")

        # Check restart policy
        policy = process_info.restart_policy
        should_restart = policy.should_restart(
            exit_code=process_info.exit_code,
            explicit_stop=False,
        )

        if not should_restart:
            return

        if process_info.restart_count >= process_info.max_restarts:
            logger.error(
                "Max restarts reached",
                process_id=process_id,
                restart_count=process_info.restart_count,
                max_restarts=process_info.max_restarts,
            )
            return

        # Calculate restart delay
        delay = process_info.restart_delay_seconds
        if policy == RestartPolicy.EXPONENTIAL_BACKOFF:
            delay = delay * (2 ** process_info.restart_count)
            delay = min(delay, 300)  # Cap at 5 minutes

        logger.info(
            "Scheduling restart",
            process_id=process_id,
            delay=delay,
            restart_count=process_info.restart_count + 1,
        )

        # Wait before restart
        await asyncio.sleep(delay)

        if self._shutdown_event.is_set():
            return

        # Restart
        await self.restart_process(process_id)

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)

                if self._shutdown_event.is_set():
                    break

                for process_id, process_info in list(self._processes.items()):
                    if process_info.state != ProcessState.RUNNING:
                        continue

                    # Update resource usage
                    agent = self._process_instances.get(process_id)
                    if agent:
                        try:
                            process_info.usage = await agent.get_resource_usage()
                            process_info.last_heartbeat = datetime.now()
                        except Exception as e:
                            logger.warning(f"Failed to get resource usage: {e}")

                    # Check resource limits
                    if self.enable_resource_monitoring:
                        exceeded, reason = process_info.usage.exceeds_limits(process_info.limits)
                        if exceeded:
                            logger.warning(
                                "Resource limit exceeded",
                                process_id=process_id,
                                reason=reason,
                            )
                            self._stats.resource_violations += 1

                            # Emit event
                            await self._emit_lifecycle_event(
                                "resource_exceeded",
                                process_id,
                                reason=reason,
                            )

                            # Terminate low-priority processes
                            if process_info.priority.value >= ProcessPriority.LOW.value:
                                await self.stop_process(process_id, graceful=True, timeout=10)

                self._stats.last_health_check = datetime.now()
                self._update_stats()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for zombie processes."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute

                if self._shutdown_event.is_set():
                    break

                now = datetime.now()

                for process_id, process_info in list(self._processes.items()):
                    # Check for zombies (stopped but task still exists)
                    if process_info.state.is_terminal():
                        if process_info.stopped_at:
                            age = (now - process_info.stopped_at).total_seconds()
                            if age > self.zombie_timeout_seconds:
                                # Clean up
                                self._process_tasks.pop(process_id, None)
                                logger.debug(f"Cleaned up zombie process: {process_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _emit_lifecycle_event(
        self,
        event_type: str,
        process_id: str,
        **kwargs,
    ) -> None:
        """Emit a process lifecycle event."""
        process_info = self._processes.get(process_id)
        if not process_info:
            return

        await self.event_bus.emit(Event(
            id=str(uuid.uuid4()),
            type=f"process.{event_type}",
            source=process_id,
            payload={
                "name": process_info.name,
                "type": process_info.type.value,
                "state": process_info.state.value,
                "priority": process_info.priority.name,
                **kwargs,
            },
        ))

    async def _emit_state_change(
        self,
        process_id: str,
        old_state: ProcessState,
        new_state: ProcessState,
    ) -> None:
        """Emit a state change event."""
        process_info = self._processes.get(process_id)
        if not process_info:
            return

        await self.event_bus.emit(Event(
            id=str(uuid.uuid4()),
            type="process.state_changed",
            source=process_id,
            payload={
                "name": process_info.name,
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        ))

        # Call state change hooks
        for hook in self._on_state_change:
            try:
                await self._call_hook(hook, process_info, old_state, new_state)
            except Exception as e:
                logger.warning(f"State change hook failed: {e}")

    async def _call_hook(self, hook: Callable, *args) -> None:
        """Call a lifecycle hook."""
        result = hook(*args)
        if asyncio.iscoroutine(result):
            await result

    async def _on_process_event(self, event: Event) -> None:
        """Handle process events."""
        # Can be used for logging, metrics, etc.
        pass

    def _update_stats(self) -> None:
        """Update supervisor statistics."""
        active = [p for p in self._processes.values() if p.state.is_active()]

        self._stats.total_processes = len(self._processes)
        self._stats.running_processes = len([p for p in active if p.state == ProcessState.RUNNING])
        self._stats.paused_processes = len([p for p in active if p.state == ProcessState.PAUSED])
        self._stats.failed_processes = len([p for p in self._processes.values() if p.state == ProcessState.FAILED])

        # Calculate totals
        self._stats.total_memory_mb = sum(p.usage.memory_mb for p in active)
        self._stats.total_tokens_used = sum(p.usage.tokens_used for p in self._processes.values())

        if self._start_time:
            self._stats.uptime_seconds = (datetime.now() - self._start_time).total_seconds()

    def _register_builtin_agents(self) -> None:
        """Register built-in agent classes."""
        try:
            from aion.systems.process.builtin_agents import (
                HealthMonitorAgent,
                GarbageCollectorAgent,
                MetricsCollectorAgent,
                WatchdogAgent,
            )

            self.register_agent_class("health_monitor", HealthMonitorAgent)
            self.register_agent_class("garbage_collector", GarbageCollectorAgent)
            self.register_agent_class("metrics_collector", MetricsCollectorAgent)
            self.register_agent_class("watchdog", WatchdogAgent)

        except ImportError as e:
            logger.warning(f"Could not import builtin agents: {e}")

    # === Query Methods ===

    def get_process(self, process_id: str) -> Optional[ProcessInfo]:
        """Get process info by ID."""
        return self._processes.get(process_id)

    def get_process_by_name(self, name: str) -> Optional[ProcessInfo]:
        """Get process info by name."""
        for p in self._processes.values():
            if p.name == name:
                return p
        return None

    def get_all_processes(self) -> List[ProcessInfo]:
        """Get all processes."""
        return list(self._processes.values())

    def get_processes_by_state(self, state: ProcessState) -> List[ProcessInfo]:
        """Get processes by state."""
        return [p for p in self._processes.values() if p.state == state]

    def get_processes_by_type(self, process_type: ProcessType) -> List[ProcessInfo]:
        """Get processes by type."""
        return [p for p in self._processes.values() if p.type == process_type]

    def get_processes_by_priority(self, priority: ProcessPriority) -> List[ProcessInfo]:
        """Get processes by priority."""
        return [p for p in self._processes.values() if p.priority == priority]

    def get_processes_by_tag(self, tag: str) -> List[ProcessInfo]:
        """Get processes by tag."""
        return [p for p in self._processes.values() if tag in p.tags]

    def get_processes_in_group(self, group: str) -> List[ProcessInfo]:
        """Get processes in a group."""
        process_ids = self._process_groups.get(group, set())
        return [self._processes[pid] for pid in process_ids if pid in self._processes]

    def get_children(self, process_id: str) -> List[ProcessInfo]:
        """Get child processes."""
        process = self._processes.get(process_id)
        if not process:
            return []
        return [
            self._processes[cid]
            for cid in process.children_ids
            if cid in self._processes
        ]

    def get_parent(self, process_id: str) -> Optional[ProcessInfo]:
        """Get parent process."""
        process = self._processes.get(process_id)
        if process and process.parent_id:
            return self._processes.get(process.parent_id)
        return None

    def get_agent_instance(self, process_id: str) -> Optional[BaseAgent]:
        """Get the agent instance for a process."""
        return self._process_instances.get(process_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get supervisor statistics."""
        self._update_stats()

        state_counts = {}
        for state in ProcessState:
            state_counts[state.value] = len(self.get_processes_by_state(state))

        return {
            **self._stats.to_dict(),
            "by_state": state_counts,
            "registered_classes": list(self._agent_classes.keys()),
            "groups": {k: len(v) for k, v in self._process_groups.items()},
        }

    # === Checkpointing ===

    async def create_checkpoint(self, process_id: str, reason: str = "manual") -> Optional[ProcessCheckpoint]:
        """Create a checkpoint for a process."""
        agent = self._process_instances.get(process_id)
        if not agent:
            return None

        checkpoint = await agent.create_checkpoint(reason)

        # Store checkpoint
        self._checkpoints[process_id].append(checkpoint)

        # Trim old checkpoints
        if len(self._checkpoints[process_id]) > self._max_checkpoints_per_process:
            self._checkpoints[process_id] = self._checkpoints[process_id][-self._max_checkpoints_per_process:]

        self._stats.checkpoints_created += 1

        return checkpoint

    def get_checkpoints(self, process_id: str) -> List[ProcessCheckpoint]:
        """Get all checkpoints for a process."""
        return self._checkpoints.get(process_id, []).copy()

    def get_latest_checkpoint(self, process_id: str) -> Optional[ProcessCheckpoint]:
        """Get the latest checkpoint for a process."""
        checkpoints = self._checkpoints.get(process_id, [])
        return checkpoints[-1] if checkpoints else None

    # === Group Operations ===

    async def stop_group(self, group: str, graceful: bool = True, timeout: float = 30.0) -> int:
        """Stop all processes in a group."""
        processes = self.get_processes_in_group(group)
        per_timeout = timeout / max(len(processes), 1)

        stopped = 0
        for process in processes:
            if await self.stop_process(process.id, graceful, per_timeout):
                stopped += 1

        return stopped

    async def pause_group(self, group: str) -> int:
        """Pause all processes in a group."""
        processes = self.get_processes_in_group(group)

        paused = 0
        for process in processes:
            if await self.pause_process(process.id):
                paused += 1

        return paused

    async def resume_group(self, group: str) -> int:
        """Resume all processes in a group."""
        processes = self.get_processes_in_group(group)

        resumed = 0
        for process in processes:
            if await self.resume_process(process.id):
                resumed += 1

        return resumed

    # === Lifecycle Hooks ===

    def on_process_start(self, callback: Callable) -> None:
        """Register a callback for process start."""
        self._on_process_start.append(callback)

    def on_process_stop(self, callback: Callable) -> None:
        """Register a callback for process stop."""
        self._on_process_stop.append(callback)

    def on_process_fail(self, callback: Callable) -> None:
        """Register a callback for process failure."""
        self._on_process_fail.append(callback)

    def on_state_change(self, callback: Callable) -> None:
        """Register a callback for state changes."""
        self._on_state_change.append(callback)

    # === Context Manager ===

    async def __aenter__(self) -> "ProcessSupervisor":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
