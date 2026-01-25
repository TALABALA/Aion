"""
AION Authentication Provider Base

Abstract base class for authentication providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

from aion.security.types import (
    AuthMethod,
    Credentials,
    User,
    MFAMethod,
)


@dataclass
class AuthProviderResult:
    """Result from an authentication provider."""

    success: bool = False
    user: Optional[User] = None
    user_id: Optional[str] = None

    # MFA
    mfa_required: bool = False
    mfa_methods: list = field(default_factory=list)
    mfa_token: Optional[str] = None

    # Error info
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Account status
    account_locked: bool = False
    lockout_until: Optional[datetime] = None
    password_expired: bool = False
    require_password_change: bool = False

    # Additional data
    claims: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthProvider(ABC):
    """
    Abstract base class for authentication providers.

    Each provider implements a specific authentication method
    (API key, JWT, password, OAuth, etc.)
    """

    @property
    @abstractmethod
    def method(self) -> AuthMethod:
        """The authentication method this provider handles."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the provider."""
        pass

    @abstractmethod
    async def authenticate(
        self,
        credentials: Credentials,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """
        Authenticate with the provided credentials.

        Args:
            credentials: The credentials to validate
            context: Optional context (IP, user agent, etc.)

        Returns:
            AuthProviderResult with success/failure details
        """
        pass

    @abstractmethod
    async def validate(
        self,
        token_or_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """
        Validate an existing token or key.

        Args:
            token_or_key: The token/key to validate
            context: Optional context

        Returns:
            AuthProviderResult with validation result
        """
        pass

    async def refresh(
        self,
        refresh_token: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """
        Refresh authentication (for providers that support it).

        Default implementation returns not supported.
        """
        return AuthProviderResult(
            success=False,
            error_code="not_supported",
            error_message=f"{self.name} does not support token refresh",
        )

    async def revoke(
        self,
        token_or_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Revoke a token or key.

        Default implementation returns False (not supported).
        """
        return False

    def can_handle(self, credentials: Credentials) -> bool:
        """Check if this provider can handle the given credentials."""
        return credentials.method == self.method


class UserRepository(Protocol):
    """Protocol for user repository (for dependency injection)."""

    async def get_by_id(self, user_id: str) -> Optional[User]:
        ...

    async def get_by_username(self, username: str) -> Optional[User]:
        ...

    async def get_by_email(self, email: str) -> Optional[User]:
        ...

    async def update(self, user: User) -> None:
        ...
