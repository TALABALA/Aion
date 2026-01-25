"""
AION Session Management

Enterprise-grade session management with security features including
concurrent session control, session elevation, and anomaly detection.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from aion.security.types import Session, User, MFAMethod

logger = structlog.get_logger(__name__)


@dataclass
class SessionConfig:
    """Session configuration."""

    # Timeouts
    session_timeout_hours: int = 24
    idle_timeout_minutes: int = 30
    elevation_timeout_minutes: int = 15

    # Limits
    max_concurrent_sessions: int = 5
    max_sessions_per_device: int = 2

    # Security
    bind_to_ip: bool = False
    bind_to_user_agent: bool = True
    require_secure_cookie: bool = True

    # Fingerprinting
    enable_device_fingerprinting: bool = True

    # Session renewal
    sliding_expiration: bool = True
    renewal_threshold_minutes: int = 60


@dataclass
class SessionSecurityEvent:
    """A security event related to a session."""

    event_type: str  # session_created, session_terminated, anomaly_detected, etc.
    session_id: str
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, critical


class SessionManager:
    """
    Manages user sessions with enterprise security features.

    Features:
    - Secure session token generation
    - Concurrent session control
    - Session elevation for sensitive operations
    - Device/IP binding
    - Anomaly detection
    - Session event notifications
    """

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
    ):
        self.config = config or SessionConfig()

        # Session storage
        self._sessions: Dict[str, Session] = {}  # session_id -> Session
        self._user_sessions: Dict[str, Set[str]] = {}  # user_id -> {session_ids}
        self._device_sessions: Dict[str, Set[str]] = {}  # device_id -> {session_ids}

        # Security events
        self._event_handlers: List[Callable[[SessionSecurityEvent], None]] = []
        self._security_events: List[SessionSecurityEvent] = []

        # Cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize session manager."""
        if self._initialized:
            return

        logger.info("Initializing Session Manager")

        # Start cleanup loop
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown session manager."""
        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    async def create_session(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
        device_fingerprint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new session.

        Enforces concurrent session limits and records security context.
        """
        # Check concurrent session limit
        await self._enforce_session_limit(user_id, device_id)

        # Generate secure session ID
        session_id = self._generate_session_id()

        # Calculate expiration
        now = datetime.now()
        expires_at = now + timedelta(hours=self.config.session_timeout_hours)

        # Create session
        session = Session(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            created_at=now,
            expires_at=expires_at,
            last_accessed_at=now,
            last_activity_at=now,
            idle_timeout_minutes=self.config.idle_timeout_minutes,
            ip_address=ip_address,
            user_agent=user_agent,
            device_id=device_id,
            device_fingerprint=device_fingerprint,
            metadata=metadata or {},
        )

        # Store session
        self._sessions[session_id] = session

        # Index by user
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = set()
        self._user_sessions[user_id].add(session_id)

        # Index by device
        if device_id:
            if device_id not in self._device_sessions:
                self._device_sessions[device_id] = set()
            self._device_sessions[device_id].add(session_id)

        # Emit event
        await self._emit_event(SessionSecurityEvent(
            event_type="session_created",
            session_id=session_id,
            user_id=user_id,
            details={
                "ip_address": ip_address,
                "user_agent": user_agent,
                "device_id": device_id,
            },
        ))

        logger.info(
            "Session created",
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at.isoformat(),
        )

        return session

    async def get_session(
        self,
        session_id: str,
        validate: bool = True,
        update_activity: bool = True,
    ) -> Optional[Session]:
        """
        Get a session by ID.

        Optionally validates and updates activity timestamp.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        if validate and not session.is_valid():
            await self._handle_invalid_session(session)
            return None

        if update_activity and session.is_active:
            session.touch()

            # Sliding expiration
            if self.config.sliding_expiration:
                await self._maybe_extend_session(session)

        return session

    async def validate_session(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> tuple[Optional[Session], Optional[str]]:
        """
        Validate a session with security checks.

        Returns (session, error_message).
        """
        session = self._sessions.get(session_id)
        if not session:
            return None, "Session not found"

        if not session.is_active:
            return None, "Session is not active"

        if not session.is_valid():
            await self._handle_invalid_session(session)
            return None, "Session has expired"

        # IP binding check
        if self.config.bind_to_ip and session.ip_address:
            if ip_address and ip_address != session.ip_address:
                await self._emit_event(SessionSecurityEvent(
                    event_type="ip_mismatch",
                    session_id=session_id,
                    user_id=session.user_id,
                    severity="warning",
                    details={
                        "expected_ip": session.ip_address,
                        "actual_ip": ip_address,
                    },
                ))
                return None, "IP address mismatch"

        # User agent binding check
        if self.config.bind_to_user_agent and session.user_agent:
            if user_agent and not self._user_agents_match(session.user_agent, user_agent):
                await self._emit_event(SessionSecurityEvent(
                    event_type="user_agent_mismatch",
                    session_id=session_id,
                    user_id=session.user_id,
                    severity="warning",
                    details={
                        "expected_ua": session.user_agent,
                        "actual_ua": user_agent,
                    },
                ))
                # Warning only, don't invalidate
                logger.warning(
                    "User agent mismatch detected",
                    session_id=session_id,
                )

        # Update activity
        session.touch()

        return session, None

    async def terminate_session(
        self,
        session_id: str,
        reason: str = "user_logout",
    ) -> bool:
        """Terminate a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.is_active = False
        session.terminated_at = datetime.now()
        session.terminated_reason = reason

        # Emit event
        await self._emit_event(SessionSecurityEvent(
            event_type="session_terminated",
            session_id=session_id,
            user_id=session.user_id,
            details={"reason": reason},
        ))

        logger.info(
            "Session terminated",
            session_id=session_id,
            user_id=session.user_id,
            reason=reason,
        )

        return True

    async def terminate_all_sessions(
        self,
        user_id: str,
        except_session_id: Optional[str] = None,
        reason: str = "user_requested",
    ) -> int:
        """Terminate all sessions for a user."""
        session_ids = self._user_sessions.get(user_id, set()).copy()
        count = 0

        for session_id in session_ids:
            if session_id != except_session_id:
                if await self.terminate_session(session_id, reason):
                    count += 1

        return count

    async def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session."""
        session = self._sessions.pop(session_id, None)
        if not session:
            return False

        # Remove from indexes
        if session.user_id in self._user_sessions:
            self._user_sessions[session.user_id].discard(session_id)

        if session.device_id and session.device_id in self._device_sessions:
            self._device_sessions[session.device_id].discard(session_id)

        return True

    # =========================================================================
    # Session Elevation
    # =========================================================================

    async def elevate_session(
        self,
        session_id: str,
        mfa_method: Optional[MFAMethod] = None,
        elevation_duration_minutes: Optional[int] = None,
    ) -> bool:
        """
        Elevate a session for sensitive operations.

        Typically called after MFA verification.
        """
        session = self._sessions.get(session_id)
        if not session or not session.is_valid():
            return False

        duration = elevation_duration_minutes or self.config.elevation_timeout_minutes
        session.is_elevated = True
        session.elevation_expires_at = datetime.now() + timedelta(minutes=duration)

        await self._emit_event(SessionSecurityEvent(
            event_type="session_elevated",
            session_id=session_id,
            user_id=session.user_id,
            details={
                "mfa_method": mfa_method.value if mfa_method else None,
                "duration_minutes": duration,
            },
        ))

        logger.info(
            "Session elevated",
            session_id=session_id,
            expires_at=session.elevation_expires_at.isoformat(),
        )

        return True

    async def revoke_elevation(self, session_id: str) -> bool:
        """Revoke session elevation."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.is_elevated = False
        session.elevation_expires_at = None

        return True

    async def require_reauthentication(
        self,
        session_id: str,
        reason: str = "security_requirement",
    ) -> bool:
        """Mark session as requiring reauthentication."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.requires_reauthentication = True
        session.is_elevated = False
        session.elevation_expires_at = None

        await self._emit_event(SessionSecurityEvent(
            event_type="reauthentication_required",
            session_id=session_id,
            user_id=session.user_id,
            severity="warning",
            details={"reason": reason},
        ))

        return True

    # =========================================================================
    # Session Queries
    # =========================================================================

    async def get_user_sessions(
        self,
        user_id: str,
        include_inactive: bool = False,
    ) -> List[Session]:
        """Get all sessions for a user."""
        session_ids = self._user_sessions.get(user_id, set())
        sessions = []

        for session_id in session_ids:
            session = self._sessions.get(session_id)
            if session:
                if include_inactive or session.is_active:
                    sessions.append(session)

        # Sort by last activity
        sessions.sort(key=lambda s: s.last_activity_at, reverse=True)

        return sessions

    async def get_active_session_count(self, user_id: str) -> int:
        """Get count of active sessions for a user."""
        sessions = await self.get_user_sessions(user_id, include_inactive=False)
        return sum(1 for s in sessions if s.is_valid())

    async def get_device_sessions(self, device_id: str) -> List[Session]:
        """Get all sessions for a device."""
        session_ids = self._device_sessions.get(device_id, set())
        sessions = []

        for session_id in session_ids:
            session = self._sessions.get(session_id)
            if session and session.is_active:
                sessions.append(session)

        return sessions

    # =========================================================================
    # Session Data
    # =========================================================================

    async def set_session_data(
        self,
        session_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """Set a value in session data."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.data[key] = value
        return True

    async def get_session_data(
        self,
        session_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a value from session data."""
        session = self._sessions.get(session_id)
        if not session:
            return default

        return session.data.get(key, default)

    async def delete_session_data(
        self,
        session_id: str,
        key: str,
    ) -> bool:
        """Delete a value from session data."""
        session = self._sessions.get(session_id)
        if not session or key not in session.data:
            return False

        del session.data[key]
        return True

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _generate_session_id(self) -> str:
        """Generate a cryptographically secure session ID."""
        return secrets.token_urlsafe(32)

    async def _enforce_session_limit(
        self,
        user_id: str,
        device_id: Optional[str] = None,
    ) -> None:
        """Enforce concurrent session limits."""
        user_session_ids = self._user_sessions.get(user_id, set())
        active_sessions = [
            self._sessions[sid]
            for sid in user_session_ids
            if sid in self._sessions and self._sessions[sid].is_active
        ]

        # Enforce per-user limit
        if len(active_sessions) >= self.config.max_concurrent_sessions:
            # Terminate oldest session
            oldest = min(active_sessions, key=lambda s: s.last_activity_at)
            await self.terminate_session(
                oldest.session_id,
                reason="concurrent_session_limit",
            )

        # Enforce per-device limit
        if device_id:
            device_session_ids = self._device_sessions.get(device_id, set())
            device_sessions = [
                self._sessions[sid]
                for sid in device_session_ids
                if sid in self._sessions and self._sessions[sid].is_active
            ]

            if len(device_sessions) >= self.config.max_sessions_per_device:
                oldest = min(device_sessions, key=lambda s: s.last_activity_at)
                await self.terminate_session(
                    oldest.session_id,
                    reason="device_session_limit",
                )

    async def _maybe_extend_session(self, session: Session) -> None:
        """Extend session expiration if using sliding window."""
        now = datetime.now()
        time_until_expiry = (session.expires_at - now).total_seconds() / 60

        if time_until_expiry < self.config.renewal_threshold_minutes:
            new_expiry = now + timedelta(hours=self.config.session_timeout_hours)
            session.expires_at = new_expiry

            logger.debug(
                "Session extended",
                session_id=session.session_id,
                new_expiry=new_expiry.isoformat(),
            )

    async def _handle_invalid_session(self, session: Session) -> None:
        """Handle an invalid/expired session."""
        if session.is_active:
            session.is_active = False
            session.terminated_at = datetime.now()
            session.terminated_reason = "expired"

            await self._emit_event(SessionSecurityEvent(
                event_type="session_expired",
                session_id=session.session_id,
                user_id=session.user_id,
            ))

    def _user_agents_match(self, expected: str, actual: str) -> bool:
        """
        Check if user agents match.

        Allows for minor variations while detecting major changes.
        """
        # Simple comparison - in production, use proper UA parsing
        expected_key = self._normalize_user_agent(expected)
        actual_key = self._normalize_user_agent(actual)
        return expected_key == actual_key

    def _normalize_user_agent(self, ua: str) -> str:
        """Normalize user agent for comparison."""
        # Extract browser family and major version
        ua_lower = ua.lower()

        if "chrome" in ua_lower:
            return "chrome"
        elif "firefox" in ua_lower:
            return "firefox"
        elif "safari" in ua_lower:
            return "safari"
        elif "edge" in ua_lower:
            return "edge"
        else:
            return hashlib.md5(ua.encode()).hexdigest()[:8]

    async def _emit_event(self, event: SessionSecurityEvent) -> None:
        """Emit a security event."""
        self._security_events.append(event)

        # Limit event history
        if len(self._security_events) > 10000:
            self._security_events = self._security_events[-5000:]

        # Notify handlers
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def add_event_handler(
        self,
        handler: Callable[[SessionSecurityEvent], None],
    ) -> None:
        """Add a security event handler."""
        self._event_handlers.append(handler)

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Every minute
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        now = datetime.now()
        expired_ids = []

        for session_id, session in self._sessions.items():
            # Keep terminated sessions for a while for auditing
            if not session.is_active:
                if session.terminated_at:
                    age = (now - session.terminated_at).total_seconds()
                    if age > 3600:  # 1 hour
                        expired_ids.append(session_id)
            elif not session.is_valid():
                await self._handle_invalid_session(session)

        for session_id in expired_ids:
            await self.delete_session(session_id)

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired sessions")

    async def get_recent_events(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionSecurityEvent]:
        """Get recent security events."""
        events = self._security_events

        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        active_sessions = sum(
            1 for s in self._sessions.values() if s.is_active and s.is_valid()
        )
        elevated_sessions = sum(
            1 for s in self._sessions.values()
            if s.is_active and s.is_elevated_valid()
        )

        return {
            "total_sessions": len(self._sessions),
            "active_sessions": active_sessions,
            "elevated_sessions": elevated_sessions,
            "unique_users": len(self._user_sessions),
            "unique_devices": len(self._device_sessions),
            "security_events": len(self._security_events),
        }
