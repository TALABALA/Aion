"""
AION API Key Authentication Provider

Handles authentication via API keys.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog

from aion.security.types import AuthMethod, Credentials, User
from aion.security.authentication.providers.base import AuthProvider, AuthProviderResult
from aion.security.authentication.tokens import TokenManager

logger = structlog.get_logger(__name__)


class APIKeyProvider(AuthProvider):
    """
    API Key authentication provider.

    Features:
    - Secure API key validation
    - Scope checking
    - IP allowlist enforcement
    - Usage tracking
    """

    def __init__(
        self,
        token_manager: TokenManager,
        user_getter: callable,
    ):
        self._token_manager = token_manager
        self._get_user = user_getter

    @property
    def method(self) -> AuthMethod:
        return AuthMethod.API_KEY

    @property
    def name(self) -> str:
        return "API Key"

    async def authenticate(
        self,
        credentials: Credentials,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """Authenticate with API key."""
        if not credentials.api_key:
            return AuthProviderResult(
                success=False,
                error_code="missing_api_key",
                error_message="API key is required",
            )

        return await self.validate(credentials.api_key, context)

    async def validate(
        self,
        token_or_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """Validate an API key."""
        context = context or {}
        client_ip = context.get("ip_address")
        required_scopes = context.get("required_scopes")

        # Validate through token manager
        token = await self._token_manager.validate_api_key(
            token_or_key,
            required_scopes=required_scopes,
            client_ip=client_ip,
        )

        if not token:
            logger.warning(
                "Invalid API key attempted",
                client_ip=client_ip,
            )
            return AuthProviderResult(
                success=False,
                error_code="invalid_api_key",
                error_message="Invalid or expired API key",
            )

        # Get user
        user = await self._get_user(token.user_id)
        if not user:
            return AuthProviderResult(
                success=False,
                error_code="user_not_found",
                error_message="User associated with API key not found",
            )

        if not user.is_active():
            return AuthProviderResult(
                success=False,
                error_code="user_inactive",
                error_message="User account is not active",
                account_locked=user.is_locked(),
                lockout_until=user.lockout_until,
            )

        return AuthProviderResult(
            success=True,
            user=user,
            user_id=user.id,
            claims={
                "token_id": token.token_id,
                "scopes": token.scopes,
            },
            metadata={
                "auth_method": "api_key",
                "token_name": token.name,
            },
        )

    async def revoke(
        self,
        token_or_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Revoke an API key by finding its token ID."""
        # Hash and find the token
        key_hash = self._token_manager._hash_token(token_or_key)
        token_id = self._token_manager._token_hash_index.get(key_hash)

        if not token_id:
            return False

        return await self._token_manager.revoke_token(
            token_id,
            reason="user_revoked",
            revoked_by=context.get("revoked_by") if context else None,
        )
