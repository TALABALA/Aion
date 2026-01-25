"""
AION JWT Authentication Provider

Handles authentication via JWT tokens.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from aion.security.types import AuthMethod, Credentials, User
from aion.security.authentication.providers.base import AuthProvider, AuthProviderResult
from aion.security.authentication.tokens import TokenManager

logger = structlog.get_logger(__name__)


class JWTProvider(AuthProvider):
    """
    JWT authentication provider.

    Features:
    - JWT token validation
    - Claims extraction
    - Audience verification
    - Token refresh support
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
        return AuthMethod.JWT

    @property
    def name(self) -> str:
        return "JWT"

    async def authenticate(
        self,
        credentials: Credentials,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """Authenticate with JWT token."""
        if not credentials.token:
            return AuthProviderResult(
                success=False,
                error_code="missing_token",
                error_message="JWT token is required",
            )

        return await self.validate(credentials.token, context)

    async def validate(
        self,
        token_or_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """Validate a JWT token."""
        context = context or {}
        required_scopes = context.get("required_scopes")

        # Validate JWT
        claims, error = await self._token_manager.validate_jwt(
            token_or_key,
            expected_type="access",
            required_scopes=required_scopes,
        )

        if error:
            error_code = "token_expired" if "expired" in error.lower() else "invalid_token"
            return AuthProviderResult(
                success=False,
                error_code=error_code,
                error_message=error,
            )

        # Get user from claims
        user_id = claims.get("sub")
        if not user_id:
            return AuthProviderResult(
                success=False,
                error_code="invalid_claims",
                error_message="Token missing subject claim",
            )

        user = await self._get_user(user_id)
        if not user:
            return AuthProviderResult(
                success=False,
                error_code="user_not_found",
                error_message="User not found",
            )

        if not user.is_active():
            return AuthProviderResult(
                success=False,
                error_code="user_inactive",
                error_message="User account is not active",
                account_locked=user.is_locked(),
            )

        return AuthProviderResult(
            success=True,
            user=user,
            user_id=user.id,
            claims=claims,
            metadata={
                "auth_method": "jwt",
                "token_jti": claims.get("jti"),
                "token_exp": claims.get("exp"),
            },
        )

    async def refresh(
        self,
        refresh_token: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """Refresh JWT tokens."""
        # Validate refresh token
        claims, error = await self._token_manager.validate_jwt(
            refresh_token,
            expected_type="refresh",
        )

        if error:
            return AuthProviderResult(
                success=False,
                error_code="invalid_refresh_token",
                error_message=error,
            )

        # Get user
        user_id = claims.get("sub")
        user = await self._get_user(user_id)

        if not user:
            return AuthProviderResult(
                success=False,
                error_code="user_not_found",
                error_message="User not found",
            )

        if not user.is_active():
            return AuthProviderResult(
                success=False,
                error_code="user_inactive",
                error_message="User account is not active",
            )

        # Generate new tokens
        access_result, new_refresh_result, error = await self._token_manager.refresh_tokens(
            refresh_token,
            user,
        )

        if error:
            return AuthProviderResult(
                success=False,
                error_code="refresh_failed",
                error_message=error,
            )

        return AuthProviderResult(
            success=True,
            user=user,
            user_id=user.id,
            claims={
                "access_token": access_result.raw_token,
                "refresh_token": new_refresh_result.raw_token,
                "expires_in": 3600,  # TODO: Get from config
            },
            metadata={
                "auth_method": "jwt_refresh",
            },
        )

    async def revoke(
        self,
        token_or_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Revoke a JWT token."""
        token_hash = self._token_manager._hash_token(token_or_key)
        token_id = self._token_manager._token_hash_index.get(token_hash)

        if not token_id:
            # Token not tracked - add to revocation list anyway
            self._token_manager._revoked_hashes.add(token_hash)
            return True

        return await self._token_manager.revoke_token(
            token_id,
            reason="user_revoked",
            revoked_by=context.get("revoked_by") if context else None,
        )
