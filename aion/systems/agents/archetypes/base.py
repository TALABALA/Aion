"""
AION Base Specialist Agent

Abstract base class for specialized agent implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

import structlog

from aion.systems.agents.types import (
    AgentInstance,
    AgentProfile,
    AgentRole,
    AgentStatus,
    TeamTask,
    Message,
    MessageType,
)

if TYPE_CHECKING:
    from aion.systems.agents.pool import AgentPool
    from aion.systems.agents.messaging import MessageBus

logger = structlog.get_logger(__name__)


class BaseSpecialist(ABC):
    """
    Abstract base class for specialist agents.

    Provides common functionality for specialized agent behavior:
    - Task processing pipeline
    - Tool invocation
    - Memory integration
    - Communication patterns
    """

    def __init__(
        self,
        instance: AgentInstance,
        pool: Optional["AgentPool"] = None,
        bus: Optional["MessageBus"] = None,
    ):
        self.instance = instance
        self.pool = pool
        self.bus = bus

        self._llm = None
        self._initialized = False

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Get the agent's role."""
        pass

    @property
    def id(self) -> str:
        """Get agent instance ID."""
        return self.instance.id

    @property
    def profile(self) -> AgentProfile:
        """Get agent profile."""
        return self.instance.profile

    @property
    def status(self) -> AgentStatus:
        """Get current status."""
        return self.instance.status

    async def initialize(self) -> None:
        """Initialize the specialist agent."""
        if self._initialized:
            return

        from aion.conversation.llm.claude import ClaudeProvider

        self._llm = ClaudeProvider()
        await self._llm.initialize()

        self._initialized = True
        logger.info(
            "Specialist initialized",
            agent_id=self.id[:8],
            role=self.role.value,
        )

    async def shutdown(self) -> None:
        """Shutdown the specialist agent."""
        if self._llm:
            await self._llm.shutdown()

        self._initialized = False

    @abstractmethod
    async def process_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Process a task according to the specialist's expertise.

        Args:
            task: Task to process
            context: Additional context

        Returns:
            Task result
        """
        pass

    async def think(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            prompt: The prompt to process
            context: Additional context

        Returns:
            Generated response
        """
        if not self._llm:
            await self.initialize()

        # Build full prompt with context
        full_prompt = prompt
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            full_prompt = f"Context:\n{context_str}\n\n{prompt}"

        response = await self._llm.complete(
            messages=[{"role": "user", "content": full_prompt}],
            system=self.profile.system_prompt,
        )

        # Extract text from response
        output = ""
        for block in response.content:
            if hasattr(block, "text"):
                output += block.text

        # Update instance stats
        self.instance.total_tokens_used += response.input_tokens + response.output_tokens
        self.instance.update_activity()

        return output

    async def handle_message(self, message: Message) -> Optional[Message]:
        """
        Handle an incoming message.

        Args:
            message: Incoming message

        Returns:
            Response message or None
        """
        if message.message_type == MessageType.TASK:
            # Process task
            task_data = message.content.get("task", {})
            if isinstance(task_data, dict):
                task = TeamTask.from_dict(task_data)
                result = await self.process_task(task, message.content)

                if message.requires_response and self.bus:
                    return await self.bus.respond(
                        message,
                        {"success": True, "result": result},
                        self.id,
                    )

        elif message.message_type == MessageType.QUERY:
            # Answer query
            response = await self.think(
                str(message.content),
                {"subject": message.subject},
            )

            if message.requires_response and self.bus:
                return await self.bus.respond(message, response, self.id)

        elif message.message_type == MessageType.FEEDBACK:
            # Process feedback
            await self._handle_feedback(message)

        return None

    async def _handle_feedback(self, message: Message) -> None:
        """Handle feedback message."""
        # Store feedback in working memory
        feedback = message.content
        if "feedback" not in self.instance.working_memory:
            self.instance.working_memory["feedback"] = []

        self.instance.working_memory["feedback"].append({
            "from": message.sender_id,
            "content": feedback,
            "timestamp": datetime.now().isoformat(),
        })

    async def send_message(
        self,
        recipient_id: str,
        content: Any,
        message_type: MessageType = MessageType.QUERY,
        subject: str = "",
    ) -> Optional[Message]:
        """
        Send a message to another agent.

        Args:
            recipient_id: Target agent ID
            content: Message content
            message_type: Type of message
            subject: Message subject

        Returns:
            Sent message or None
        """
        if not self.bus:
            return None

        return await self.bus.send(
            sender_id=self.id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            subject=subject,
        )

    async def broadcast(
        self,
        content: Any,
        subject: str = "",
        team_id: Optional[str] = None,
    ) -> Optional[Message]:
        """
        Broadcast a message.

        Args:
            content: Message content
            subject: Message subject
            team_id: Limit to team

        Returns:
            Broadcast message or None
        """
        if not self.bus:
            return None

        return await self.bus.broadcast(
            sender_id=self.id,
            content=content,
            subject=subject,
            team_id=team_id or self.instance.current_team_id,
        )

    def remember(self, key: str, value: Any) -> None:
        """Store something in working memory."""
        self.instance.working_memory[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        """Recall something from working memory."""
        return self.instance.working_memory.get(key, default)

    def update_status(self, status: AgentStatus) -> None:
        """Update agent status."""
        self.instance.status = status
        self.instance.update_activity()

        if self.pool:
            self.pool.update_status(self.id, status)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "role": self.role.value,
            "name": self.profile.name,
            "status": self.status.value,
            "tasks_completed": self.instance.tasks_completed,
            "tokens_used": self.instance.total_tokens_used,
        }
