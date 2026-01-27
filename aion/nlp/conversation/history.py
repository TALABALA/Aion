"""
AION Conversation History - Track message history for sessions.

Provides conversation context management with summarization
for long-running sessions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from aion.nlp.types import ConversationMessage, ProgrammingSession

logger = structlog.get_logger(__name__)


class ConversationHistory:
    """
    Manages conversation history for programming sessions.

    Features:
    - Message storage and retrieval
    - Context window management
    - History search
    """

    def __init__(self, max_messages: int = 200):
        self._max_messages = max_messages

    def add_message(
        self,
        session: ProgrammingSession,
        role: str,
        content: str,
        **metadata: Any,
    ) -> ConversationMessage:
        """Add a message to session history."""
        msg = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata,
        )
        session.messages.append(msg)

        # Trim if over limit
        if len(session.messages) > self._max_messages:
            session.messages = session.messages[-self._max_messages:]

        return msg

    def get_context(
        self,
        session: ProgrammingSession,
        max_messages: int = 20,
    ) -> List[Dict[str, str]]:
        """Get recent context for LLM prompts."""
        recent = session.messages[-max_messages:]
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent
        ]

    def search(
        self,
        session: ProgrammingSession,
        query: str,
    ) -> List[ConversationMessage]:
        """Search history for relevant messages."""
        query_lower = query.lower()
        return [
            msg for msg in session.messages
            if query_lower in msg.content.lower()
        ]

    def get_summary(self, session: ProgrammingSession) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        user_msgs = [m for m in session.messages if m.role == "user"]
        assistant_msgs = [m for m in session.messages if m.role == "assistant"]

        return {
            "total_messages": len(session.messages),
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "iterations": session.iterations,
            "duration_seconds": session.duration_seconds,
        }
