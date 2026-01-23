"""
AION Session Manager

Manages conversation sessions with:
- In-memory caching
- Persistence integration
- Session expiry and cleanup
- Concurrent access handling
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
import uuid

import structlog

from aion.conversation.types import Conversation, ConversationStatus

logger = structlog.get_logger(__name__)


class SessionManager:
    """
    Manages conversation sessions.

    Features:
    - In-memory session cache with LRU eviction
    - Automatic session expiry
    - Optional persistence integration
    - Thread-safe concurrent access
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        session_ttl_hours: int = 24,
        persistence: Any = None,
    ):
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.persistence = persistence

        self._sessions: dict[str, Conversation] = {}
        self._last_access: dict[str, datetime] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the session manager."""
        if self._initialized:
            return

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._initialized = True
        logger.info(
            "Session manager initialized",
            max_sessions=self.max_sessions,
            ttl_hours=self.session_ttl.total_seconds() / 3600,
        )

    async def shutdown(self) -> None:
        """Shutdown the session manager."""
        logger.info("Shutting down session manager")

        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.persistence:
            for conversation in self._sessions.values():
                await self._persist_conversation(conversation)

        self._sessions.clear()
        self._last_access.clear()
        self._locks.clear()

        self._initialized = False
        logger.info("Session manager shutdown complete")

    async def save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation to the session."""
        async with self._global_lock:
            self._sessions[conversation.id] = conversation
            self._last_access[conversation.id] = datetime.now()

            if conversation.id not in self._locks:
                self._locks[conversation.id] = asyncio.Lock()

        if self.persistence:
            await self._persist_conversation(conversation)

        if len(self._sessions) > self.max_sessions:
            await self._evict_oldest()

        logger.debug("Saved conversation", conversation_id=conversation.id[:8])

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        if conversation_id in self._sessions:
            self._last_access[conversation_id] = datetime.now()
            return self._sessions[conversation_id]

        if self.persistence:
            conversation = await self._load_conversation(conversation_id)
            if conversation:
                async with self._global_lock:
                    self._sessions[conversation_id] = conversation
                    self._last_access[conversation_id] = datetime.now()
                    self._locks[conversation_id] = asyncio.Lock()
                return conversation

        return None

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        async with self._global_lock:
            if conversation_id in self._sessions:
                self._sessions[conversation_id].status = ConversationStatus.DELETED
                del self._sessions[conversation_id]
            self._last_access.pop(conversation_id, None)
            self._locks.pop(conversation_id, None)

        if self.persistence:
            await self._delete_from_persistence(conversation_id)

        logger.info("Deleted conversation", conversation_id=conversation_id[:8])
        return True

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        status: Optional[ConversationStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """List conversations with optional filtering."""
        conversations = list(self._sessions.values())

        if user_id:
            conversations = [c for c in conversations if c.user_id == user_id]

        if status:
            conversations = [c for c in conversations if c.status == status]

        conversations = [
            c for c in conversations if c.status != ConversationStatus.DELETED
        ]

        conversations.sort(key=lambda c: c.updated_at, reverse=True)

        return conversations[offset : offset + limit]

    async def get_conversation_lock(self, conversation_id: str) -> asyncio.Lock:
        """Get a lock for a specific conversation."""
        if conversation_id not in self._locks:
            async with self._global_lock:
                if conversation_id not in self._locks:
                    self._locks[conversation_id] = asyncio.Lock()
        return self._locks[conversation_id]

    def active_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        now = datetime.now()
        active_recent = sum(
            1
            for last_access in self._last_access.values()
            if now - last_access < timedelta(hours=1)
        )

        return {
            "total_sessions": len(self._sessions),
            "active_last_hour": active_recent,
            "max_sessions": self.max_sessions,
            "ttl_hours": self.session_ttl.total_seconds() / 3600,
        }

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up expired sessions."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)

                now = datetime.now()
                expired = []

                for conv_id, last_access in list(self._last_access.items()):
                    if now - last_access > self.session_ttl:
                        expired.append(conv_id)

                for conv_id in expired:
                    if self.persistence and conv_id in self._sessions:
                        await self._persist_conversation(self._sessions[conv_id])

                    async with self._global_lock:
                        self._sessions.pop(conv_id, None)
                        self._last_access.pop(conv_id, None)
                        self._locks.pop(conv_id, None)

                if expired:
                    logger.info(f"Cleaned up {len(expired)} expired sessions")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    async def _evict_oldest(self) -> None:
        """Evict oldest sessions to stay under limit."""
        if len(self._sessions) <= self.max_sessions:
            return

        sorted_ids = sorted(
            self._last_access.keys(),
            key=lambda x: self._last_access[x],
        )

        to_evict = len(self._sessions) - self.max_sessions + 10

        for conv_id in sorted_ids[:to_evict]:
            if self.persistence and conv_id in self._sessions:
                await self._persist_conversation(self._sessions[conv_id])

            async with self._global_lock:
                self._sessions.pop(conv_id, None)
                self._last_access.pop(conv_id, None)
                self._locks.pop(conv_id, None)

        logger.info(f"Evicted {to_evict} sessions")

    async def _persist_conversation(self, conversation: Conversation) -> None:
        """Persist conversation to storage."""
        if not self.persistence:
            return

        try:
            if hasattr(self.persistence, "save_state"):
                await self.persistence.save_state(
                    f"conversation:{conversation.id}",
                    conversation.to_full_dict(),
                )
        except Exception as e:
            logger.error(f"Failed to persist conversation: {e}")

    async def _load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load conversation from storage."""
        if not self.persistence:
            return None

        try:
            if hasattr(self.persistence, "load_state"):
                data = await self.persistence.load_state(f"conversation:{conversation_id}")
                if data:
                    return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")

        return None

    async def _delete_from_persistence(self, conversation_id: str) -> None:
        """Delete conversation from storage."""
        if not self.persistence:
            return

        try:
            if hasattr(self.persistence, "delete_state"):
                await self.persistence.delete_state(f"conversation:{conversation_id}")
        except Exception as e:
            logger.error(f"Failed to delete conversation from persistence: {e}")


class SessionContext:
    """
    Context manager for working with a conversation session.

    Provides automatic locking and saving.
    """

    def __init__(
        self,
        manager: SessionManager,
        conversation_id: str,
        auto_save: bool = True,
    ):
        self.manager = manager
        self.conversation_id = conversation_id
        self.auto_save = auto_save
        self._conversation: Optional[Conversation] = None
        self._lock: Optional[asyncio.Lock] = None

    async def __aenter__(self) -> Conversation:
        self._lock = await self.manager.get_conversation_lock(self.conversation_id)
        await self._lock.acquire()

        self._conversation = await self.manager.get_conversation(self.conversation_id)
        if not self._conversation:
            raise ValueError(f"Conversation not found: {self.conversation_id}")

        return self._conversation

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if self.auto_save and self._conversation and exc_type is None:
                await self.manager.save_conversation(self._conversation)
        finally:
            if self._lock:
                self._lock.release()
