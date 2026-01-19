"""
AION Base Agent

State-of-the-art abstract base class for all AION agents with:
- Full lifecycle management (initialize, run, pause, resume, shutdown)
- Communication primitives (messaging, events, channels)
- Resource tracking and limits enforcement
- State management and checkpointing
- Tool access and execution tracking
- Child process spawning
- Heartbeat and health reporting
"""

from __future__ import annotations

import abc
import asyncio
import uuid
import traceback
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Callable, TypeVar, Generic
from collections import deque
from contextlib import asynccontextmanager

import structlog

from aion.systems.process.models import (
    AgentConfig,
    AgentMessage,
    Event,
    ResourceUsage,
    ProcessState,
    ProcessCheckpoint,
    MessageType,
    SignalType,
)

if TYPE_CHECKING:
    from aion.systems.process.supervisor import ProcessSupervisor
    from aion.systems.process.event_bus import EventBus

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class AgentContext:
    """
    Context for agent execution providing access to AION systems.
    Passed to agents during initialization for dependency injection.
    """

    def __init__(
        self,
        process_id: str,
        config: AgentConfig,
        supervisor: "ProcessSupervisor",
        event_bus: "EventBus",
        kernel: Optional[Any] = None,
    ):
        self.process_id = process_id
        self.config = config
        self.supervisor = supervisor
        self.event_bus = event_bus
        self.kernel = kernel

    @property
    def memory(self):
        """Access AION memory system if available."""
        if self.kernel and hasattr(self.kernel, "memory"):
            return self.kernel.memory
        return None

    @property
    def tools(self):
        """Access AION tool orchestrator if available."""
        if self.kernel and hasattr(self.kernel, "tools"):
            return self.kernel.tools
        return None

    @property
    def planning(self):
        """Access AION planning system if available."""
        if self.kernel and hasattr(self.kernel, "planning"):
            return self.kernel.planning
        return None

    @property
    def llm(self):
        """Access AION LLM adapter if available."""
        if self.kernel and hasattr(self.kernel, "llm"):
            return self.kernel.llm
        return None


class BaseAgent(abc.ABC):
    """
    Abstract base class for AION agents.

    All custom agents should extend this class and implement:
    - run(): Main agent execution loop
    - (optional) on_message(): Handle incoming messages
    - (optional) on_signal(): Handle control signals

    The base class provides:
    - Lifecycle management (initialize, shutdown, pause, resume)
    - Communication (send/receive messages, emit events)
    - Resource tracking (tokens, tools, memory)
    - State management (get/set state, checkpointing)
    - Child process spawning
    """

    def __init__(
        self,
        process_id: str,
        config: AgentConfig,
        supervisor: "ProcessSupervisor",
        event_bus: "EventBus",
        kernel: Optional[Any] = None,
    ):
        self.process_id = process_id
        self.config = config
        self.supervisor = supervisor
        self.event_bus = event_bus
        self.kernel = kernel

        # Create context for convenience
        self.context = AgentContext(
            process_id=process_id,
            config=config,
            supervisor=supervisor,
            event_bus=event_bus,
            kernel=kernel,
        )

        # State management
        self._state = ProcessState.CREATED
        self._paused = asyncio.Event()
        self._paused.set()  # Not paused initially
        self._shutdown_requested = False
        self._shutdown_complete = asyncio.Event()

        # Communication
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._subscriptions: list[str] = []  # Subscription IDs
        self._pending_responses: dict[str, asyncio.Future[AgentMessage]] = {}
        self._sequence_number: int = 0

        # Resource tracking
        self._started_at: Optional[datetime] = None
        self._tokens_used: int = 0
        self._tokens_this_minute: int = 0
        self._minute_start: datetime = datetime.now()
        self._tool_calls: int = 0
        self._errors: int = 0
        self._last_activity: datetime = datetime.now()

        # Internal state (agent-specific persistent state)
        self._internal_state: dict[str, Any] = {}

        # Checkpointing
        self._last_checkpoint: Optional[datetime] = None
        self._checkpoint_interval: int = config.checkpoint_interval_seconds

        # Message history for context
        self._message_history: deque[AgentMessage] = deque(maxlen=100)

        # Heartbeat
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: float = 30.0

        # Logger with context
        self.logger = logger.bind(
            agent_id=process_id,
            agent_name=config.name,
            agent_class=config.agent_class,
        )

    # === Lifecycle Methods ===

    async def initialize(self) -> None:
        """
        Initialize the agent (called before run).
        Sets up communication channels and calls custom initialization.
        """
        self.logger.info("Initializing agent")
        self._state = ProcessState.STARTING

        try:
            # Subscribe to configured input channels
            for channel in self.config.input_channels:
                await self._subscribe_channel(channel)

            # Subscribe to broadcast channels
            for channel in self.config.broadcast_channels:
                await self._subscribe_channel(channel)

            # Subscribe to direct messages for this agent
            await self._subscribe_channel(f"agent.{self.process_id}")
            await self._subscribe_channel(f"agent.{self.config.name}")

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Record start time
            self._started_at = datetime.now()

            # Call custom initialization
            await self.on_initialize()

            self._state = ProcessState.RUNNING
            self.logger.info("Agent initialized successfully")

        except Exception as e:
            self._state = ProcessState.FAILED
            self.logger.error("Agent initialization failed", error=str(e))
            raise

    async def on_initialize(self) -> None:
        """Override for custom initialization logic."""
        pass

    @abc.abstractmethod
    async def run(self) -> None:
        """
        Main agent execution loop.

        This method should:
        1. Check self._shutdown_requested periodically
        2. Await self._paused.wait() to respect pause state
        3. Process messages from self.get_next_message()
        4. Perform agent-specific work

        Example:
            async def run(self):
                while not self._shutdown_requested:
                    await self._paused.wait()

                    # Process messages
                    message = await self.get_next_message(timeout=1.0)
                    if message:
                        await self.process_message(message)

                    # Do agent work
                    await self.do_work()
        """
        pass

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        self.logger.info("Shutting down agent")
        self._shutdown_requested = True
        self._state = ProcessState.STOPPING

        # Unblock if paused
        self._paused.set()

        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cancel pending responses
        for correlation_id, future in list(self._pending_responses.items()):
            if not future.done():
                future.cancel()
        self._pending_responses.clear()

        # Create final checkpoint
        if self.config.auto_checkpoint:
            await self.create_checkpoint("pre_shutdown")

        # Call custom cleanup
        await self.on_shutdown()

        # Unsubscribe from all channels
        for sub_id in self._subscriptions[:]:
            await self.event_bus.unsubscribe_by_id(sub_id)
        self._subscriptions.clear()

        self._state = ProcessState.STOPPED
        self._shutdown_complete.set()
        self.logger.info("Agent shutdown complete")

    async def on_shutdown(self) -> None:
        """Override for custom shutdown logic."""
        pass

    async def pause(self) -> None:
        """Pause the agent."""
        self._paused.clear()
        self._state = ProcessState.PAUSED
        await self.on_pause()
        self.logger.info("Agent paused")

    async def on_pause(self) -> None:
        """Override for custom pause logic."""
        pass

    async def resume(self) -> None:
        """Resume the agent from paused state."""
        await self.on_resume()
        self._paused.set()
        self._state = ProcessState.RUNNING
        self.logger.info("Agent resumed")

    async def on_resume(self) -> None:
        """Override for custom resume logic."""
        pass

    async def receive_signal(self, signal: SignalType, payload: dict) -> None:
        """Handle an incoming control signal."""
        self.logger.debug("Received signal", signal=signal.value if isinstance(signal, SignalType) else signal)

        if isinstance(signal, str):
            try:
                signal = SignalType(signal)
            except ValueError:
                signal = SignalType.CUSTOM

        if signal == SignalType.PAUSE:
            await self.pause()
        elif signal == SignalType.RESUME:
            await self.resume()
        elif signal == SignalType.STOP:
            await self.shutdown()
        elif signal == SignalType.CHECKPOINT:
            await self.create_checkpoint("signal")
        elif signal == SignalType.RELOAD:
            await self.on_reload(payload)
        else:
            await self.on_signal(signal, payload)

    async def on_signal(self, signal: SignalType, payload: dict) -> None:
        """Override to handle custom signals."""
        pass

    async def on_reload(self, payload: dict) -> None:
        """Override to handle reload signal (e.g., config refresh)."""
        pass

    # === Communication Methods ===

    async def _subscribe_channel(self, channel: str) -> None:
        """Subscribe to an event channel."""
        sub_id = await self.event_bus.subscribe(
            pattern=channel,
            handler=self._on_event,
            subscriber_id=self.process_id,
        )
        self._subscriptions.append(sub_id)

    async def _on_event(self, event: Event) -> None:
        """Internal event handler - converts events to messages."""
        # Extract message content
        content = event.payload.get("content", "")
        message_type_str = event.payload.get("message_type", "event")

        try:
            message_type = MessageType(message_type_str)
        except ValueError:
            message_type = MessageType.EVENT

        message = AgentMessage(
            id=event.id,
            from_agent=event.source,
            to_agent=self.process_id,
            content=str(content),
            message_type=message_type,
            payload=event.payload,
            timestamp=event.timestamp,
            correlation_id=event.correlation_id,
        )

        # Check if this is a response to a pending request
        if event.correlation_id and event.correlation_id in self._pending_responses:
            future = self._pending_responses.pop(event.correlation_id)
            if not future.done():
                future.set_result(message)
            return

        # Add to message queue
        await self._message_queue.put(message)

        # Update activity
        self._last_activity = datetime.now()

        # Store in history
        self._message_history.append(message)

    async def send_message(
        self,
        to_agent: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        payload: Optional[dict] = None,
        require_response: bool = False,
        timeout: float = 30.0,
    ) -> Optional[AgentMessage]:
        """
        Send a message to another agent.

        Args:
            to_agent: Target agent ID, name, or "broadcast"
            content: Message content
            message_type: Type of message
            payload: Additional data
            require_response: If True, wait for response
            timeout: Response timeout in seconds

        Returns:
            Response message if require_response=True, else None
        """
        self._sequence_number += 1
        message_id = str(uuid.uuid4())
        correlation_id = message_id if require_response else None

        message = AgentMessage(
            id=message_id,
            from_agent=self.process_id,
            to_agent=to_agent,
            content=content,
            message_type=message_type,
            payload=payload or {},
            requires_response=require_response,
            response_timeout=timeout,
            correlation_id=correlation_id,
            sequence_number=self._sequence_number,
        )

        # Determine channel
        if to_agent == "broadcast":
            channel = "agent.broadcast"
        else:
            channel = f"agent.{to_agent}"

        # Create event
        event = Event(
            id=message_id,
            type=channel,
            source=self.process_id,
            payload={
                "content": content,
                "message_type": message_type.value,
                "from_agent": self.process_id,
                "from_name": self.config.name,
                **message.payload,
            },
            correlation_id=correlation_id,
        )

        if require_response:
            # Create future for response
            future: asyncio.Future[AgentMessage] = asyncio.get_event_loop().create_future()
            self._pending_responses[correlation_id] = future

            try:
                await self.event_bus.emit(event)
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                self._pending_responses.pop(correlation_id, None)
                self.logger.warning("Message response timeout", to_agent=to_agent)
                return None
        else:
            await self.event_bus.emit(event)
            return None

    async def reply_to(
        self,
        message: AgentMessage,
        content: str,
        payload: Optional[dict] = None,
    ) -> None:
        """Reply to a received message."""
        response = message.create_response(
            from_agent=self.process_id,
            content=content,
            payload=payload,
        )

        event = Event(
            id=response.id,
            type=f"agent.{message.from_agent}",
            source=self.process_id,
            payload={
                "content": content,
                "message_type": MessageType.RESPONSE.value,
                "from_agent": self.process_id,
                "from_name": self.config.name,
                **(payload or {}),
            },
            correlation_id=response.correlation_id,
        )

        await self.event_bus.emit(event)

    async def emit_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        to_channels: Optional[list[str]] = None,
    ) -> None:
        """Emit an event to configured output channels."""
        channels = to_channels or self.config.output_channels

        for channel in channels:
            full_channel = f"{channel}.{event_type}" if not event_type.startswith(channel) else event_type

            await self.event_bus.emit(Event(
                id=str(uuid.uuid4()),
                type=full_channel,
                source=self.process_id,
                payload={
                    "event_type": event_type,
                    "agent_id": self.process_id,
                    "agent_name": self.config.name,
                    **payload,
                },
            ))

    async def get_next_message(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[AgentMessage]:
        """
        Get the next message from the queue.

        Args:
            timeout: Timeout in seconds (None = block forever)

        Returns:
            Next message or None on timeout
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=timeout,
                )
            else:
                return await self._message_queue.get()
        except asyncio.TimeoutError:
            return None

    def has_messages(self) -> bool:
        """Check if there are pending messages."""
        return not self._message_queue.empty()

    def message_count(self) -> int:
        """Get number of pending messages."""
        return self._message_queue.qsize()

    def get_recent_messages(self, limit: int = 10) -> list[AgentMessage]:
        """Get recent message history."""
        return list(self._message_history)[-limit:]

    # === Child Process Management ===

    async def spawn_child(self, config: AgentConfig) -> str:
        """
        Spawn a child agent process.

        Args:
            config: Configuration for the child agent

        Returns:
            Process ID of the spawned child
        """
        return await self.supervisor.spawn_agent(config, parent_id=self.process_id)

    async def stop_child(self, child_id: str, graceful: bool = True) -> bool:
        """Stop a child agent."""
        child = self.supervisor.get_process(child_id)
        if child and child.parent_id == self.process_id:
            return await self.supervisor.stop_process(child_id, graceful=graceful)
        return False

    async def send_signal_to_child(self, child_id: str, signal: SignalType, payload: Optional[dict] = None) -> bool:
        """Send a signal to a child process."""
        child = self.supervisor.get_process(child_id)
        if child and child.parent_id == self.process_id:
            return await self.supervisor.send_signal(child_id, signal, payload or {})
        return False

    def get_children(self) -> list:
        """Get all child process info."""
        return self.supervisor.get_children(self.process_id)

    def get_active_children_count(self) -> int:
        """Get count of active child processes."""
        children = self.get_children()
        return len([c for c in children if c.state.is_active()])

    # === Resource Tracking ===

    async def get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        runtime = 0.0
        if self._started_at:
            runtime = (datetime.now() - self._started_at).total_seconds()

        # Calculate tokens per minute
        now = datetime.now()
        if (now - self._minute_start).total_seconds() >= 60:
            self._tokens_this_minute = 0
            self._minute_start = now

        return ResourceUsage(
            tokens_used=self._tokens_used,
            tokens_per_minute=self._tokens_this_minute,
            runtime_seconds=runtime,
            tool_calls=self._tool_calls,
            active_children=self.get_active_children_count(),
            queue_size=self._message_queue.qsize(),
            events_emitted=0,  # Could track if needed
            events_received=len(self._message_history),
            errors_count=self._errors,
            last_activity=self._last_activity,
        )

    def record_token_usage(self, tokens: int) -> None:
        """Record token usage."""
        self._tokens_used += tokens
        self._tokens_this_minute += tokens
        self._last_activity = datetime.now()

    def record_tool_call(self) -> None:
        """Record a tool call."""
        self._tool_calls += 1
        self._last_activity = datetime.now()

    def record_error(self) -> None:
        """Record an error."""
        self._errors += 1

    # === State Management ===

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get internal state value."""
        return self._internal_state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set internal state value."""
        self._internal_state[key] = value
        self._last_activity = datetime.now()

    def update_state(self, updates: dict[str, Any]) -> None:
        """Update multiple state values."""
        self._internal_state.update(updates)
        self._last_activity = datetime.now()

    def delete_state(self, key: str) -> bool:
        """Delete a state key."""
        if key in self._internal_state:
            del self._internal_state[key]
            return True
        return False

    def get_all_state(self) -> dict[str, Any]:
        """Get all internal state."""
        return self._internal_state.copy()

    def clear_state(self) -> None:
        """Clear all internal state."""
        self._internal_state.clear()

    # === Checkpointing ===

    async def create_checkpoint(self, reason: str = "manual") -> ProcessCheckpoint:
        """Create a checkpoint of current state."""
        # Serialize message queue
        queue_snapshot = []
        temp_messages = []

        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                temp_messages.append(msg)
                queue_snapshot.append(msg.to_dict())
            except asyncio.QueueEmpty:
                break

        # Restore queue
        for msg in temp_messages:
            await self._message_queue.put(msg)

        checkpoint = ProcessCheckpoint(
            id=str(uuid.uuid4()),
            process_id=self.process_id,
            timestamp=datetime.now(),
            state=self._state,
            internal_state=self._internal_state.copy(),
            message_queue_snapshot=queue_snapshot,
            resource_usage=await self.get_resource_usage(),
            restart_count=0,  # Will be set by supervisor
            reason=reason,
            metadata={
                "agent_name": self.config.name,
                "agent_class": self.config.agent_class,
            },
        )

        self._last_checkpoint = datetime.now()

        # Emit checkpoint event
        await self.emit_event("checkpoint_created", {
            "checkpoint_id": checkpoint.id,
            "reason": reason,
        })

        return checkpoint

    async def restore_from_checkpoint(self, checkpoint: ProcessCheckpoint) -> None:
        """Restore state from a checkpoint."""
        self._internal_state = checkpoint.internal_state.copy()

        # Restore message queue
        for msg_dict in checkpoint.message_queue_snapshot:
            msg = AgentMessage.from_dict(msg_dict)
            await self._message_queue.put(msg)

        self.logger.info(
            "Restored from checkpoint",
            checkpoint_id=checkpoint.id,
            messages_restored=len(checkpoint.message_queue_snapshot),
        )

    # === Heartbeat ===

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                if self._shutdown_requested:
                    break

                # Emit heartbeat
                await self.event_bus.emit(Event(
                    id=str(uuid.uuid4()),
                    type="agent.heartbeat",
                    source=self.process_id,
                    payload={
                        "agent_name": self.config.name,
                        "state": self._state.value,
                        "queue_size": self._message_queue.qsize(),
                        "uptime_seconds": (datetime.now() - self._started_at).total_seconds() if self._started_at else 0,
                    },
                ))

                # Auto-checkpoint if needed
                if self.config.auto_checkpoint:
                    if self._last_checkpoint is None or \
                       (datetime.now() - self._last_checkpoint).total_seconds() > self._checkpoint_interval:
                        await self.create_checkpoint("scheduled")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")

    # === Tool Execution Helpers ===

    @asynccontextmanager
    async def tool_execution(self, tool_name: str):
        """Context manager for tracking tool execution."""
        self.record_tool_call()
        start_time = datetime.now()

        try:
            yield
        except Exception as e:
            self.record_error()
            self.logger.error(f"Tool execution failed: {tool_name}", error=str(e))
            raise
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Tool execution complete: {tool_name}", duration=duration)

    async def execute_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Execute a tool through AION's tool orchestrator."""
        if not self.context.tools:
            raise RuntimeError("Tool orchestrator not available")

        async with self.tool_execution(tool_name):
            return await self.context.tools.execute_tool(tool_name, params)

    # === Memory Helpers ===

    async def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Store a memory in AION's memory system."""
        if not self.context.memory:
            return None

        return await self.context.memory.store(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata={
                "agent_id": self.process_id,
                "agent_name": self.config.name,
                **(metadata or {}),
            },
        )

    async def search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
    ) -> list:
        """Search AION's memory system."""
        if not self.context.memory:
            return []

        return await self.context.memory.search(
            query=query,
            limit=limit,
            memory_type=memory_type,
        )

    # === LLM Helpers ===

    async def think(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Use LLM for agent reasoning."""
        if not self.context.llm:
            raise RuntimeError("LLM adapter not available")

        messages = []

        # Add system prompt if configured
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt,
            })

        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"Context:\n{context}",
            })

        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = await self.context.llm.complete(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Track token usage
        if hasattr(response, "usage") and response.usage:
            total_tokens = response.usage.get("total_tokens", 0)
            self.record_token_usage(total_tokens)

        return response.content

    # === Utility Methods ===

    def is_running(self) -> bool:
        """Check if agent is in running state."""
        return self._state == ProcessState.RUNNING and not self._shutdown_requested

    def is_paused(self) -> bool:
        """Check if agent is paused."""
        return self._state == ProcessState.PAUSED

    async def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to complete."""
        try:
            await asyncio.wait_for(self._shutdown_complete.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def sleep(self, seconds: float) -> bool:
        """
        Sleep that respects shutdown requests.
        Returns False if shutdown was requested during sleep.
        """
        try:
            await asyncio.wait_for(
                asyncio.create_task(self._wait_for_shutdown_or_timeout(seconds)),
                timeout=seconds + 0.1,
            )
            return not self._shutdown_requested
        except asyncio.TimeoutError:
            return not self._shutdown_requested

    async def _wait_for_shutdown_or_timeout(self, seconds: float) -> None:
        """Helper for sleep that checks shutdown."""
        end_time = datetime.now() + timedelta(seconds=seconds)
        while datetime.now() < end_time and not self._shutdown_requested:
            await asyncio.sleep(min(0.1, seconds))
