"""
AION Session Manager - Manage NLP programming sessions.

Handles session lifecycle, cleanup, and state management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog

from aion.nlp.types import ProgrammingSession

logger = structlog.get_logger(__name__)


class SessionManager:
    """
    Manages NLP programming sessions.

    Handles:
    - Session creation and retrieval
    - Idle session cleanup
    - Session state persistence
    - Concurrent session limits
    """

    def __init__(
        self,
        max_sessions: int = 100,
        idle_timeout_seconds: float = 1800.0,
    ):
        self._sessions: Dict[str, ProgrammingSession] = {}
        self._max_sessions = max_sessions
        self._idle_timeout = idle_timeout_seconds
        self._lock = asyncio.Lock()

    def create(self, user_id: str) -> ProgrammingSession:
        """Create a new programming session."""
        # Check capacity
        if len(self._sessions) >= self._max_sessions:
            self._cleanup_stale()
            if len(self._sessions) >= self._max_sessions:
                # Remove oldest session
                oldest = min(self._sessions.values(), key=lambda s: s.last_activity)
                del self._sessions[oldest.id]

        session = ProgrammingSession(user_id=user_id)
        self._sessions[session.id] = session

        logger.debug("Session created", session_id=session.id, user_id=user_id)
        return session

    def get(self, session_id: str) -> Optional[ProgrammingSession]:
        """Get a session by ID."""
        session = self._sessions.get(session_id)
        if session:
            session.last_activity = datetime.now(timezone.utc)
        return session

    def get_or_create(
        self,
        session_id: Optional[str],
        user_id: str,
    ) -> ProgrammingSession:
        """Get existing session or create new one."""
        if session_id:
            session = self.get(session_id)
            if session:
                return session

        return self.create(user_id)

    def list_user_sessions(self, user_id: str) -> List[ProgrammingSession]:
        """List all sessions for a user."""
        return [
            s for s in self._sessions.values()
            if s.user_id == user_id and s.state == "active"
        ]

    def end_session(self, session_id: str) -> bool:
        """End a programming session."""
        session = self._sessions.get(session_id)
        if session:
            session.state = "completed"
            logger.debug("Session ended", session_id=session_id)
            return True
        return False

    def _cleanup_stale(self) -> int:
        """Remove stale sessions. Returns count removed."""
        now = datetime.now(timezone.utc)
        stale = [
            sid for sid, session in self._sessions.items()
            if (now - session.last_activity).total_seconds() > self._idle_timeout
        ]
        for sid in stale:
            self._sessions[sid].state = "abandoned"
            del self._sessions[sid]

        if stale:
            logger.info("Cleaned up stale sessions", count=len(stale))
        return len(stale)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.state == "active")

    @property
    def total_count(self) -> int:
        return len(self._sessions)
