"""
AION Password Authentication Provider

Handles username/password authentication with security features.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog

from aion.security.types import AuthMethod, Credentials, User, UserStatus
from aion.security.authentication.providers.base import AuthProvider, AuthProviderResult

logger = structlog.get_logger(__name__)


class PasswordPolicy:
    """Password policy configuration."""

    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 128,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_numbers: bool = True,
        require_special: bool = True,
        special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?",
        max_age_days: int = 90,
        history_count: int = 5,
        lockout_threshold: int = 5,
        lockout_duration_minutes: int = 30,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_numbers = require_numbers
        self.require_special = require_special
        self.special_chars = special_chars
        self.max_age_days = max_age_days
        self.history_count = history_count
        self.lockout_threshold = lockout_threshold
        self.lockout_duration_minutes = lockout_duration_minutes

    def validate(self, password: str) -> tuple[bool, str]:
        """Validate a password against the policy."""
        if len(password) < self.min_length:
            return False, f"Password must be at least {self.min_length} characters"

        if len(password) > self.max_length:
            return False, f"Password must be at most {self.max_length} characters"

        if self.require_uppercase and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"

        if self.require_lowercase and not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"

        if self.require_numbers and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"

        if self.require_special and not any(c in self.special_chars for c in password):
            return False, f"Password must contain at least one special character"

        return True, ""


class PasswordProvider(AuthProvider):
    """
    Password authentication provider.

    Features:
    - Secure password hashing (argon2id)
    - Account lockout after failed attempts
    - Password expiration
    - MFA trigger
    - Brute force protection
    """

    def __init__(
        self,
        user_getter: callable,
        user_updater: callable,
        password_policy: Optional[PasswordPolicy] = None,
    ):
        self._get_user = user_getter
        self._update_user = user_updater
        self.policy = password_policy or PasswordPolicy()

        # Track failed attempts (in-memory, use Redis in production)
        self._failed_attempts: Dict[str, list] = {}  # ip -> [(timestamp, username)]

    @property
    def method(self) -> AuthMethod:
        return AuthMethod.BASIC

    @property
    def name(self) -> str:
        return "Password"

    async def authenticate(
        self,
        credentials: Credentials,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """Authenticate with username and password."""
        context = context or {}
        username = credentials.username
        password = credentials.password
        ip_address = context.get("ip_address")

        if not username or not password:
            return AuthProviderResult(
                success=False,
                error_code="missing_credentials",
                error_message="Username and password are required",
            )

        # Check for brute force
        if ip_address and self._is_ip_blocked(ip_address):
            logger.warning(
                "Blocked login attempt from rate-limited IP",
                ip=ip_address,
            )
            return AuthProviderResult(
                success=False,
                error_code="too_many_attempts",
                error_message="Too many failed attempts. Please try again later.",
            )

        # Get user
        user = await self._get_user_by_identifier(username)

        if not user:
            # Don't reveal if user exists
            await self._record_failed_attempt(ip_address, username)
            return AuthProviderResult(
                success=False,
                error_code="invalid_credentials",
                error_message="Invalid username or password",
            )

        # Check if account is locked
        if user.is_locked():
            return AuthProviderResult(
                success=False,
                error_code="account_locked",
                error_message="Account is temporarily locked",
                account_locked=True,
                lockout_until=user.lockout_until,
            )

        # Check if account is active
        if user.status == UserStatus.SUSPENDED:
            return AuthProviderResult(
                success=False,
                error_code="account_suspended",
                error_message="Account has been suspended",
            )

        if user.status == UserStatus.PENDING_VERIFICATION:
            return AuthProviderResult(
                success=False,
                error_code="email_not_verified",
                error_message="Please verify your email address",
            )

        # Verify password
        if not self._verify_password(password, user.password_hash):
            await self._handle_failed_login(user, ip_address)
            return AuthProviderResult(
                success=False,
                error_code="invalid_credentials",
                error_message="Invalid username or password",
                account_locked=user.is_locked(),
                lockout_until=user.lockout_until,
            )

        # Clear failed attempts on success
        user.failed_login_attempts = 0
        user.last_failed_login_at = None
        user.lockout_until = None

        # Check password expiration
        if self._is_password_expired(user):
            return AuthProviderResult(
                success=True,
                user=user,
                user_id=user.id,
                password_expired=True,
                require_password_change=True,
                error_code="password_expired",
                error_message="Your password has expired. Please change it.",
            )

        # Check if MFA is required
        if user.mfa.enabled:
            mfa_token = secrets.token_urlsafe(32)
            return AuthProviderResult(
                success=True,
                user=user,
                user_id=user.id,
                mfa_required=True,
                mfa_methods=user.mfa.methods,
                mfa_token=mfa_token,
            )

        # Update last login
        user.last_login_at = datetime.now()
        user.last_active_at = datetime.now()
        await self._update_user(user)

        return AuthProviderResult(
            success=True,
            user=user,
            user_id=user.id,
            metadata={
                "auth_method": "password",
            },
        )

    async def validate(
        self,
        token_or_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """Password provider doesn't support token validation."""
        return AuthProviderResult(
            success=False,
            error_code="not_supported",
            error_message="Password authentication does not use tokens",
        )

    async def change_password(
        self,
        user: User,
        current_password: str,
        new_password: str,
    ) -> tuple[bool, str]:
        """Change a user's password."""
        # Verify current password
        if not self._verify_password(current_password, user.password_hash):
            return False, "Current password is incorrect"

        # Validate new password
        is_valid, error = self.policy.validate(new_password)
        if not is_valid:
            return False, error

        # Check password history (if implemented)
        # ...

        # Hash and update
        user.password_hash = self._hash_password(new_password)
        user.password_changed_at = datetime.now()
        user.password_expires_at = datetime.now() + timedelta(days=self.policy.max_age_days)
        user.require_password_change = False
        user.updated_at = datetime.now()

        await self._update_user(user)

        logger.info("Password changed", user_id=user.id)

        return True, ""

    async def reset_password(
        self,
        user: User,
        new_password: str,
    ) -> tuple[bool, str]:
        """Reset a user's password (admin action)."""
        # Validate new password
        is_valid, error = self.policy.validate(new_password)
        if not is_valid:
            return False, error

        # Hash and update
        user.password_hash = self._hash_password(new_password)
        user.password_changed_at = datetime.now()
        user.password_expires_at = datetime.now() + timedelta(days=self.policy.max_age_days)
        user.require_password_change = True  # Force change on next login
        user.failed_login_attempts = 0
        user.lockout_until = None
        user.updated_at = datetime.now()

        await self._update_user(user)

        logger.info("Password reset", user_id=user.id)

        return True, ""

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _get_user_by_identifier(self, identifier: str) -> Optional[User]:
        """Get user by username or email."""
        # Try as username first
        user = await self._get_user(identifier, by="username")
        if user:
            return user

        # Try as email
        if "@" in identifier:
            user = await self._get_user(identifier, by="email")

        return user

    def _hash_password(self, password: str) -> str:
        """Hash a password using argon2id."""
        try:
            import argon2

            ph = argon2.PasswordHasher(
                time_cost=3,
                memory_cost=65536,
                parallelism=4,
                hash_len=32,
                salt_len=16,
            )
            return ph.hash(password)
        except ImportError:
            # Fallback to bcrypt
            import bcrypt

            return bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        if not password_hash:
            return False

        try:
            if password_hash.startswith("$argon2"):
                import argon2

                ph = argon2.PasswordHasher()
                try:
                    ph.verify(password_hash, password)
                    return True
                except argon2.exceptions.VerifyMismatchError:
                    return False
            else:
                # bcrypt hash
                import bcrypt

                return bcrypt.checkpw(password.encode(), password_hash.encode())
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def _is_password_expired(self, user: User) -> bool:
        """Check if user's password has expired."""
        if not user.password_expires_at:
            return False
        return datetime.now() > user.password_expires_at

    async def _handle_failed_login(self, user: User, ip_address: Optional[str]) -> None:
        """Handle a failed login attempt."""
        user.failed_login_attempts += 1
        user.last_failed_login_at = datetime.now()

        # Record attempt
        await self._record_failed_attempt(ip_address, user.username)

        # Lock account if threshold exceeded
        if user.failed_login_attempts >= self.policy.lockout_threshold:
            user.lockout_until = datetime.now() + timedelta(
                minutes=self.policy.lockout_duration_minutes
            )
            user.status = UserStatus.LOCKED

            logger.warning(
                "Account locked due to failed attempts",
                user_id=user.id,
                attempts=user.failed_login_attempts,
            )

        await self._update_user(user)

    async def _record_failed_attempt(
        self,
        ip_address: Optional[str],
        username: str,
    ) -> None:
        """Record a failed login attempt for rate limiting."""
        if not ip_address:
            return

        now = datetime.now()

        if ip_address not in self._failed_attempts:
            self._failed_attempts[ip_address] = []

        # Add attempt
        self._failed_attempts[ip_address].append((now, username))

        # Clean old attempts (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        self._failed_attempts[ip_address] = [
            (ts, u) for ts, u in self._failed_attempts[ip_address] if ts > cutoff
        ]

    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if an IP is blocked due to too many failed attempts."""
        if ip_address not in self._failed_attempts:
            return False

        now = datetime.now()
        cutoff = now - timedelta(minutes=15)

        recent_attempts = [
            (ts, u) for ts, u in self._failed_attempts[ip_address] if ts > cutoff
        ]

        # Block if more than 20 attempts in 15 minutes
        return len(recent_attempts) > 20
