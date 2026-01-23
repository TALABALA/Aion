"""
AION Conversation Types

Core dataclasses for the conversational interface.
Provides type-safe representations of messages, conversations, and streaming events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Literal, Optional, Union
import uuid


class MessageRole(str, Enum):
    """Role of a message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(str, Enum):
    """Type of message content."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    PLAN = "plan"


class ConversationStatus(str, Enum):
    """Status of a conversation."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class StreamEventType(str, Enum):
    """Types of streaming events."""
    TEXT = "text"
    THINKING = "thinking"
    THINKING_START = "thinking_start"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_INPUT = "tool_use_input"
    TOOL_USE_END = "tool_use_end"
    TOOL_EXECUTING = "tool_executing"
    TOOL_RESULT = "tool_result"
    CONVERSATION_CREATED = "conversation_created"
    DONE = "done"
    ERROR = "error"


@dataclass
class TextContent:
    """Text content block."""
    type: Literal["text"] = "text"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextContent":
        return cls(text=data.get("text", ""))


@dataclass
class ImageContent:
    """Image content block."""
    type: Literal["image"] = "image"
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageContent":
        return cls(source=data.get("source", {}))


@dataclass
class AudioContent:
    """Audio content block."""
    type: Literal["audio"] = "audio"
    source: dict[str, Any] = field(default_factory=dict)
    transcript: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"type": self.type, "source": self.source}
        if self.transcript:
            result["transcript"] = self.transcript
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioContent":
        return cls(
            source=data.get("source", {}),
            transcript=data.get("transcript"),
        )


@dataclass
class FileContent:
    """File content block."""
    type: Literal["file"] = "file"
    name: str = ""
    path: str = ""
    mime_type: str = ""
    size: int = 0
    content: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "type": self.type,
            "name": self.name,
            "path": self.path,
            "mime_type": self.mime_type,
            "size": self.size,
        }
        if self.content:
            result["content"] = self.content
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileContent":
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            mime_type=data.get("mime_type", ""),
            size=data.get("size", 0),
            content=data.get("content"),
        )


@dataclass
class ToolUseContent:
    """Tool use request content."""
    type: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolUseContent":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=data.get("input", {}),
        )


@dataclass
class ToolResultContent:
    """Tool result content."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultContent":
        return cls(
            tool_use_id=data.get("tool_use_id", ""),
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
        )


@dataclass
class ThinkingContent:
    """Chain-of-thought thinking block."""
    type: Literal["thinking"] = "thinking"
    thinking: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "thinking": self.thinking}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThinkingContent":
        return cls(thinking=data.get("thinking", ""))


@dataclass
class PlanContent:
    """Execution plan content block."""
    type: Literal["plan"] = "plan"
    plan_id: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "plan_id": self.plan_id,
            "steps": self.steps,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanContent":
        return cls(
            plan_id=data.get("plan_id", ""),
            steps=data.get("steps", []),
            status=data.get("status", "pending"),
        )


ContentBlock = Union[
    TextContent,
    ImageContent,
    AudioContent,
    FileContent,
    ToolUseContent,
    ToolResultContent,
    ThinkingContent,
    PlanContent,
]


def content_block_from_dict(data: dict[str, Any]) -> ContentBlock:
    """Create a content block from a dictionary."""
    block_type = data.get("type", "text")

    type_map = {
        "text": TextContent,
        "image": ImageContent,
        "audio": AudioContent,
        "file": FileContent,
        "tool_use": ToolUseContent,
        "tool_result": ToolResultContent,
        "thinking": ThinkingContent,
        "plan": PlanContent,
    }

    cls = type_map.get(block_type, TextContent)
    return cls.from_dict(data)


@dataclass
class Message:
    """A message in a conversation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: list[ContentBlock] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    tool_use_id: Optional[str] = None

    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "role": self.role.value,
            "content": [c.to_dict() for c in self.content],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
        if self.tool_use_id:
            d["tool_use_id"] = self.tool_use_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        content = [
            content_block_from_dict(c)
            for c in data.get("content", [])
        ]

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=MessageRole(data.get("role", "user")),
            content=content,
            created_at=created_at,
            metadata=data.get("metadata", {}),
            tool_use_id=data.get("tool_use_id"),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> "Message":
        """Create a user message."""
        return cls(
            role=MessageRole.USER,
            content=[TextContent(text=text)],
            **kwargs,
        )

    @classmethod
    def assistant(cls, text: str, **kwargs: Any) -> "Message":
        """Create an assistant message."""
        return cls(
            role=MessageRole.ASSISTANT,
            content=[TextContent(text=text)],
            **kwargs,
        )

    @classmethod
    def system(cls, text: str, **kwargs: Any) -> "Message":
        """Create a system message."""
        return cls(
            role=MessageRole.SYSTEM,
            content=[TextContent(text=text)],
            **kwargs,
        )

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
        **kwargs: Any,
    ) -> "Message":
        """Create a tool result message."""
        return cls(
            role=MessageRole.USER,
            content=[ToolResultContent(
                tool_use_id=tool_use_id,
                content=content,
                is_error=is_error,
            )],
            tool_use_id=tool_use_id,
            **kwargs,
        )

    def get_text(self) -> str:
        """Get all text content concatenated."""
        texts = []
        for block in self.content:
            if isinstance(block, TextContent):
                texts.append(block.text)
            elif isinstance(block, ThinkingContent):
                texts.append(f"[Thinking: {block.thinking}]")
            elif isinstance(block, ToolUseContent):
                texts.append(f"[Tool: {block.name}]")
            elif isinstance(block, ToolResultContent):
                texts.append(f"[Result: {block.content[:100]}...]")
        return "\n".join(texts)

    def get_tool_uses(self) -> list[ToolUseContent]:
        """Get all tool use content blocks."""
        return [
            block for block in self.content
            if isinstance(block, ToolUseContent)
        ]

    def has_tool_use(self) -> bool:
        """Check if message contains tool use."""
        return any(isinstance(block, ToolUseContent) for block in self.content)

    def add_content(self, content: ContentBlock) -> None:
        """Add a content block to the message."""
        self.content.append(content)


@dataclass
class ConversationConfig:
    """Configuration for a conversation."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: Optional[float] = None

    system_prompt: Optional[str] = None

    max_context_messages: int = 50
    max_context_tokens: int = 100000
    include_system_context: bool = True

    memory_enabled: bool = True
    memory_retrieval_count: int = 5
    memory_auto_store: bool = True
    memory_importance_threshold: float = 0.5

    tools_enabled: bool = True
    allowed_tools: Optional[list[str]] = None
    max_tool_iterations: int = 10
    tool_timeout_seconds: float = 30.0

    planning_enabled: bool = True
    auto_plan_complex_tasks: bool = True

    streaming_enabled: bool = True

    content_filter_enabled: bool = True

    extended_thinking_enabled: bool = False
    thinking_budget_tokens: int = 10000

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system_prompt": self.system_prompt,
            "max_context_messages": self.max_context_messages,
            "max_context_tokens": self.max_context_tokens,
            "include_system_context": self.include_system_context,
            "memory_enabled": self.memory_enabled,
            "memory_retrieval_count": self.memory_retrieval_count,
            "memory_auto_store": self.memory_auto_store,
            "tools_enabled": self.tools_enabled,
            "allowed_tools": self.allowed_tools,
            "max_tool_iterations": self.max_tool_iterations,
            "planning_enabled": self.planning_enabled,
            "auto_plan_complex_tasks": self.auto_plan_complex_tasks,
            "streaming_enabled": self.streaming_enabled,
            "content_filter_enabled": self.content_filter_enabled,
            "extended_thinking_enabled": self.extended_thinking_enabled,
            "thinking_budget_tokens": self.thinking_budget_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationConfig":
        return cls(
            model=data.get("model", "claude-sonnet-4-20250514"),
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p"),
            system_prompt=data.get("system_prompt"),
            max_context_messages=data.get("max_context_messages", 50),
            max_context_tokens=data.get("max_context_tokens", 100000),
            include_system_context=data.get("include_system_context", True),
            memory_enabled=data.get("memory_enabled", True),
            memory_retrieval_count=data.get("memory_retrieval_count", 5),
            memory_auto_store=data.get("memory_auto_store", True),
            tools_enabled=data.get("tools_enabled", True),
            allowed_tools=data.get("allowed_tools"),
            max_tool_iterations=data.get("max_tool_iterations", 10),
            planning_enabled=data.get("planning_enabled", True),
            auto_plan_complex_tasks=data.get("auto_plan_complex_tasks", True),
            streaming_enabled=data.get("streaming_enabled", True),
            content_filter_enabled=data.get("content_filter_enabled", True),
            extended_thinking_enabled=data.get("extended_thinking_enabled", False),
            thinking_budget_tokens=data.get("thinking_budget_tokens", 10000),
        )


@dataclass
class Conversation:
    """A conversation session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: ConversationConfig = field(default_factory=ConversationConfig)

    messages: list[Message] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    title: Optional[str] = None
    user_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    status: ConversationStatus = ConversationStatus.ACTIVE

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    message_count: int = 0

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.message_count += 1
        self.total_input_tokens += message.input_tokens
        self.total_output_tokens += message.output_tokens
        self.updated_at = datetime.now()

    def get_context_messages(self, limit: Optional[int] = None) -> list[Message]:
        """Get messages within context window."""
        max_messages = limit or self.config.max_context_messages
        return self.messages[-max_messages:]

    def get_last_user_message(self) -> Optional[Message]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the last assistant message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg
        return None

    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.message_count = 0
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "config": self.config.to_dict(),
            "metadata": self.metadata,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Convert to dict including all messages."""
        result = self.to_dict()
        result["messages"] = [m.to_dict() for m in self.messages]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        config = ConversationConfig.from_dict(data.get("config", {}))

        messages = [
            Message.from_dict(m)
            for m in data.get("messages", [])
        ]

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        status = data.get("status", "active")
        if isinstance(status, str):
            status = ConversationStatus(status)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            config=config,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            title=data.get("title"),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
            status=status,
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            message_count=data.get("message_count", len(messages)),
        )


@dataclass
class StreamEvent:
    """An event in a streaming response."""
    type: str
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def text(cls, text: str) -> "StreamEvent":
        return cls(type=StreamEventType.TEXT.value, data=text)

    @classmethod
    def thinking(cls, thinking: str) -> "StreamEvent":
        return cls(type=StreamEventType.THINKING.value, data=thinking)

    @classmethod
    def tool_use_start(cls, tool_id: str, name: str) -> "StreamEvent":
        return cls(
            type=StreamEventType.TOOL_USE_START.value,
            data={"id": tool_id, "name": name},
        )

    @classmethod
    def tool_result(
        cls,
        tool_name: str,
        result: str,
        is_error: bool = False,
    ) -> "StreamEvent":
        return cls(
            type=StreamEventType.TOOL_RESULT.value,
            data={"tool": tool_name, "result": result, "is_error": is_error},
        )

    @classmethod
    def done(cls, data: Optional[dict] = None) -> "StreamEvent":
        return cls(type=StreamEventType.DONE.value, data=data)

    @classmethod
    def error(cls, error: str) -> "StreamEvent":
        return cls(type=StreamEventType.ERROR.value, data=error)


@dataclass
class ConversationRequest:
    """A request to the conversation system."""
    message: str
    conversation_id: Optional[str] = None

    images: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    audio: Optional[dict[str, Any]] = None

    config_overrides: dict[str, Any] = field(default_factory=dict)

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "conversation_id": self.conversation_id,
            "images": self.images,
            "files": self.files,
            "audio": self.audio,
            "config_overrides": self.config_overrides,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


@dataclass
class ConversationResponse:
    """A response from the conversation system."""
    message: Message
    conversation_id: str

    tools_used: list[str] = field(default_factory=list)
    memories_retrieved: int = 0
    plan_executed: bool = False

    input_tokens: int = 0
    output_tokens: int = 0

    latency_ms: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message.to_dict(),
            "conversation_id": self.conversation_id,
            "tools_used": self.tools_used,
            "memories_retrieved": self.memories_retrieved,
            "plan_executed": self.plan_executed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class ConversationStats:
    """Statistics for the conversation system."""
    conversations_created: int = 0
    messages_processed: int = 0
    tool_calls: int = 0
    memory_retrievals: int = 0
    memory_stores: int = 0
    errors: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    active_sessions: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversations_created": self.conversations_created,
            "messages_processed": self.messages_processed,
            "tool_calls": self.tool_calls,
            "memory_retrievals": self.memory_retrievals,
            "memory_stores": self.memory_stores,
            "errors": self.errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "active_sessions": self.active_sessions,
            "avg_latency_ms": self.avg_latency_ms,
        }
