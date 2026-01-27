"""
AION Conversation Context - Track and manage conversation state.

Maintains a rich context window that enables multi-turn
programming sessions with coherent understanding.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.nlp.types import (
    ConversationMessage,
    DeployedSystem,
    Intent,
    ProgrammingSession,
)

logger = structlog.get_logger(__name__)


class ConversationContext:
    """
    Manages conversation context for multi-turn NLP programming.

    Tracks:
    - Recent messages and intents
    - Referenced systems and entities
    - User preferences and patterns
    - Active session state
    """

    def __init__(self, max_context_messages: int = 20):
        self._max_messages = max_context_messages
        self._sessions: Dict[str, ProgrammingSession] = {}

    def get_or_create_session(
        self,
        session_id: Optional[str],
        user_id: str,
    ) -> ProgrammingSession:
        """Get existing session or create a new one."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_activity = datetime.now()
            return session

        session = ProgrammingSession(user_id=user_id)
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ProgrammingSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def build_context(self, session: ProgrammingSession) -> Dict[str, Any]:
        """Build a context dict from session state for LLM prompts."""
        context: Dict[str, Any] = {}

        # Recent conversation
        recent = session.get_context_window(self._max_messages)
        if recent:
            context["recent_messages"] = [
                {"role": m.role, "content": m.content[:200]}
                for m in recent[-5:]
            ]

        # Current state
        if session.current_intent:
            context["current_intent"] = {
                "type": session.current_intent.type.value,
                "name": session.current_intent.name,
            }

        if session.current_spec:
            spec = session.current_spec
            context["current_spec"] = {
                "name": getattr(spec, "name", "unknown"),
                "type": type(spec).__name__,
            }

        # Referenced systems
        if session.referenced_systems:
            context["referenced_systems"] = session.referenced_systems

        # Iteration count
        context["iteration"] = session.iterations

        # Session context variables
        context.update(session.context)

        return context

    def extract_references(
        self,
        text: str,
        session: ProgrammingSession,
    ) -> List[str]:
        """Extract references to existing systems from text."""
        references: List[str] = []

        # Check for pronouns referencing current work
        pronoun_patterns = ["it", "that", "this", "the same", "the tool", "the workflow", "the agent"]
        text_lower = text.lower()

        for pronoun in pronoun_patterns:
            if pronoun in text_lower and session.current_intent:
                name = session.current_intent.name
                if name:
                    references.append(name)
                break

        # Check for explicit name references from history
        for intent in session.intent_history:
            if intent.name and intent.name.lower() in text_lower:
                references.append(intent.name)

        return references

    def cleanup_stale_sessions(self, max_idle_seconds: float = 1800.0) -> int:
        """Remove stale sessions. Returns count of removed sessions."""
        now = datetime.now()
        stale_ids = []

        for sid, session in self._sessions.items():
            idle = (now - session.last_activity).total_seconds()
            if idle > max_idle_seconds:
                stale_ids.append(sid)

        for sid in stale_ids:
            del self._sessions[sid]

        if stale_ids:
            logger.info("Cleaned up stale sessions", count=len(stale_ids))

        return len(stale_ids)

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)
