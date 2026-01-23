"""
AION Conversation REST API

REST endpoints for the conversation interface.
"""

from __future__ import annotations

from typing import Any, Optional
import json

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from aion.conversation.manager import ConversationManager
from aion.conversation.types import (
    ConversationRequest,
    ConversationConfig,
)
from aion.conversation.streaming.events import sse_generator

logger = structlog.get_logger(__name__)


class ChatRequest(BaseModel):
    """Chat request body."""
    message: str = Field(..., min_length=1, description="The user's message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    images: list[dict] = Field(default_factory=list, description="Image attachments")
    files: list[dict] = Field(default_factory=list, description="File attachments")
    config: dict = Field(default_factory=dict, description="Config overrides")
    user_id: Optional[str] = Field(None, description="User ID")
    metadata: dict = Field(default_factory=dict, description="Request metadata")


class CreateConversationRequest(BaseModel):
    """Create conversation request."""
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    model: Optional[str] = Field(None, description="Model to use")
    title: Optional[str] = Field(None, description="Conversation title")
    user_id: Optional[str] = Field(None, description="User ID")
    config: dict = Field(default_factory=dict, description="Configuration options")
    metadata: dict = Field(default_factory=dict, description="Metadata")


class UpdateConversationRequest(BaseModel):
    """Update conversation request."""
    title: Optional[str] = None
    metadata: Optional[dict] = None
    config: Optional[dict] = None


class ConversationResponse(BaseModel):
    """Conversation response."""
    id: str
    title: Optional[str]
    user_id: Optional[str]
    status: str
    created_at: str
    updated_at: str
    message_count: int
    total_tokens: int


class MessageResponse(BaseModel):
    """Message response."""
    id: str
    role: str
    content: list[dict]
    created_at: str


class ChatResponse(BaseModel):
    """Chat response."""
    conversation_id: str
    message: dict
    tools_used: list[str]
    memories_retrieved: int
    input_tokens: int
    output_tokens: int
    latency_ms: float


def create_conversation_router(manager: ConversationManager) -> APIRouter:
    """Create the conversation API router."""

    router = APIRouter(prefix="/conversations", tags=["conversations"])

    @router.post("", response_model=ConversationResponse)
    async def create_conversation(request: CreateConversationRequest):
        """Create a new conversation."""
        config_dict = {
            **({"system_prompt": request.system_prompt} if request.system_prompt else {}),
            **({"model": request.model} if request.model else {}),
            **request.config,
        }

        config = ConversationConfig(**config_dict) if config_dict else None

        conversation = await manager.create_conversation(
            config=config,
            user_id=request.user_id,
            title=request.title,
            metadata=request.metadata,
        )

        return conversation.to_dict()

    @router.get("", response_model=list[ConversationResponse])
    async def list_conversations(
        user_id: Optional[str] = Query(None, description="Filter by user ID"),
        limit: int = Query(50, ge=1, le=100, description="Max conversations to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
    ):
        """List conversations."""
        conversations = await manager.list_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        return [c.to_dict() for c in conversations]

    @router.get("/{conversation_id}", response_model=ConversationResponse)
    async def get_conversation(conversation_id: str):
        """Get conversation details."""
        conversation = await manager.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return conversation.to_dict()

    @router.patch("/{conversation_id}", response_model=ConversationResponse)
    async def update_conversation(
        conversation_id: str,
        request: UpdateConversationRequest,
    ):
        """Update conversation metadata."""
        conversation = await manager.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if request.title is not None:
            conversation.title = request.title

        if request.metadata is not None:
            conversation.metadata.update(request.metadata)

        if request.config is not None:
            for key, value in request.config.items():
                if hasattr(conversation.config, key):
                    setattr(conversation.config, key, value)

        await manager.sessions.save_conversation(conversation)

        return conversation.to_dict()

    @router.delete("/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        """Delete a conversation."""
        success = await manager.delete_conversation(conversation_id)

        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"success": True, "conversation_id": conversation_id}

    @router.get("/{conversation_id}/messages")
    async def get_messages(
        conversation_id: str,
        limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
    ):
        """Get conversation messages."""
        conversation = await manager.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = conversation.messages[offset : offset + limit]

        return {
            "messages": [m.to_dict() for m in messages],
            "total": len(conversation.messages),
            "conversation_id": conversation_id,
        }

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Send a chat message and get response."""
        try:
            conv_request = ConversationRequest(
                message=request.message,
                conversation_id=request.conversation_id,
                images=request.images,
                files=request.files,
                config_overrides=request.config,
                user_id=request.user_id,
                metadata=request.metadata,
            )

            response = await manager.chat(conv_request)

            return {
                "conversation_id": response.conversation_id,
                "message": response.message.to_dict(),
                "tools_used": response.tools_used,
                "memories_retrieved": response.memories_retrieved,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """Send a chat message and stream response."""
        conv_request = ConversationRequest(
            message=request.message,
            conversation_id=request.conversation_id,
            images=request.images,
            files=request.files,
            config_overrides=request.config,
            user_id=request.user_id,
            metadata=request.metadata,
        )

        async def event_generator():
            async for sse_data in sse_generator(
                manager.chat_stream(conv_request),
                include_event_names=True,
            ):
                yield sse_data

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post("/{conversation_id}/messages")
    async def add_message(
        conversation_id: str,
        request: ChatRequest,
    ):
        """Add a message to an existing conversation."""
        conversation = await manager.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conv_request = ConversationRequest(
            message=request.message,
            conversation_id=conversation_id,
            images=request.images,
            files=request.files,
            config_overrides=request.config,
            user_id=request.user_id,
            metadata=request.metadata,
        )

        response = await manager.chat(conv_request)

        return {
            "conversation_id": response.conversation_id,
            "message": response.message.to_dict(),
            "tools_used": response.tools_used,
            "memories_retrieved": response.memories_retrieved,
            "latency_ms": response.latency_ms,
        }

    @router.post("/{conversation_id}/clear")
    async def clear_messages(conversation_id: str):
        """Clear all messages from a conversation."""
        conversation = await manager.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation.clear_messages()
        await manager.sessions.save_conversation(conversation)

        return {"success": True, "conversation_id": conversation_id}

    @router.get("/stats")
    async def get_stats():
        """Get conversation system statistics."""
        return manager.get_stats()

    @router.get("/{conversation_id}/export")
    async def export_conversation(
        conversation_id: str,
        format: str = Query("json", description="Export format (json, markdown)"),
    ):
        """Export a conversation."""
        conversation = await manager.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if format == "json":
            return conversation.to_full_dict()

        elif format == "markdown":
            lines = [f"# {conversation.title or 'Conversation'}\n"]
            lines.append(f"Created: {conversation.created_at.isoformat()}\n")
            lines.append("---\n")

            for msg in conversation.messages:
                role = "**User:**" if msg.role.value == "user" else "**Assistant:**"
                lines.append(f"\n{role}\n\n{msg.get_text()}\n")

            return {"content": "\n".join(lines), "format": "markdown"}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown format: {format}")

    return router


def create_health_router(manager: ConversationManager) -> APIRouter:
    """Create health check router."""

    router = APIRouter(tags=["health"])

    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy" if manager.is_initialized else "initializing",
            "active_sessions": manager.sessions.active_count(),
        }

    @router.get("/ready")
    async def readiness_check():
        """Readiness check endpoint."""
        if not manager.is_initialized:
            raise HTTPException(status_code=503, detail="Service not ready")

        return {"status": "ready"}

    return router
