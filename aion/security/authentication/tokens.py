"""
AION Token Management

Comprehensive token management for API keys, JWT tokens, and refresh tokens.
Includes secure generation, validation, rotation, and revocation.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.security.types import (
    AuthToken,
    TokenStatus,
    TokenType,
    User,
)

logger = structlog.get_logger(__name__)


# Token format: aion_{prefix}_{random_part}
TOKEN_PREFIX = "aion"
API_KEY_PREFIX_LENGTH = 8
API_KEY_SECRET_LENGTH = 32


@dataclass
class TokenGenerationResult:
    """Result of token generation."""

    raw_token: str  # The actual token value (show to user once)
    token_record: AuthToken  # The stored record (hash only)


class TokenManager:
    """
    Manages authentication tokens with enterprise-grade security.

    Features:
    - Secure token generation with cryptographic randomness
    - Constant-time token comparison to prevent timing attacks
    - Token rotation with grace periods
    - Automatic expiration handling
    - Token family tracking for refresh token rotation
    - Prefix-based token identification
    """

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        access_token_expiry_minutes: int = 60,
        refresh_token_expiry_days: int = 30,
        api_key_expiry_days: Optional[int] = None,
        rotation_grace_period_seconds: int = 60,
    ):
        self.jwt_secret = jwt_secret or secrets.token_hex(64)
        self.jwt_algorithm = jwt_algorithm
        self.access_token_expiry = timedelta(minutes=access_token_expiry_minutes)
        self.refresh_token_expiry = timedelta(days=refresh_token_expiry_days)
        self.api_key_expiry = (
            timedelta(days=api_key_expiry_days) if api_key_expiry_days else None
        )
        self.rotation_grace_period = timedelta(seconds=rotation_grace_period_seconds)

        # Token storage (in production, use database)
        self._tokens: Dict[str, AuthToken] = {}  # token_id -> token
        self._token_hash_index: Dict[str, str] = {}  # hash -> token_id
        self._user_tokens: Dict[str, List[str]] = {}  # user_id -> [token_ids]
        self._refresh_families: Dict[str, List[str]] = {}  # family_id -> [token_ids]

        # Revoked tokens cache (for quick lookup)
        self._revoked_hashes: set = set()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize token manager."""
        if self._initialized:
            return

        logger.info("Initializing Token Manager")

        # Start cleanup loop
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown token manager."""
        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    # =========================================================================
    # API Key Management
    # =========================================================================

    async def create_api_key(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        name: str = "",
        description: str = "",
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenGenerationResult:
        """
        Create a new API key.

        Returns the raw key (shown once) and the stored record.
        """
        # Generate secure random key
        prefix = secrets.token_hex(API_KEY_PREFIX_LENGTH // 2)
        secret_part = secrets.token_urlsafe(API_KEY_SECRET_LENGTH)
        raw_key = f"{TOKEN_PREFIX}_{prefix}_{secret_part}"

        # Hash for storage
        key_hash = self._hash_token(raw_key)

        # Calculate expiry
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        elif self.api_key_expiry:
            expires_at = datetime.now() + self.api_key_expiry

        # Create token record
        token = AuthToken(
            token_type=TokenType.API_KEY,
            token_hash=key_hash,
            token_prefix=f"{TOKEN_PREFIX}_{prefix}",
            user_id=user_id,
            tenant_id=tenant_id,
            name=name,
            description=description,
            scopes=scopes or [],
            expires_at=expires_at,
            allowed_ips=allowed_ips or [],
        )

        # Store
        self._store_token(token)

        logger.info(
            "Created API key",
            token_id=token.token_id,
            user_id=user_id,
            name=name,
            expires_at=expires_at.isoformat() if expires_at else None,
        )

        return TokenGenerationResult(raw_token=raw_key, token_record=token)

    async def validate_api_key(
        self,
        api_key: str,
        required_scopes: Optional[List[str]] = None,
        client_ip: Optional[str] = None,
    ) -> Optional[AuthToken]:
        """
        Validate an API key.

        Returns the token record if valid, None otherwise.
        """
        if not api_key:
            return None

        # Check format
        if not api_key.startswith(f"{TOKEN_PREFIX}_"):
            return None

        # Hash and lookup
        key_hash = self._hash_token(api_key)

        # Check revocation cache first (fast path)
        if key_hash in self._revoked_hashes:
            return None

        token_id = self._token_hash_index.get(key_hash)
        if not token_id:
            return None

        token = self._tokens.get(token_id)
        if not token:
            return None

        # Validate token
        if not token.is_valid():
            if token.expires_at and datetime.now() > token.expires_at:
                token.status = TokenStatus.EXPIRED
            return None

        # Check scopes
        if required_scopes:
            if not self._has_required_scopes(token.scopes, required_scopes):
                logger.warning(
                    "API key missing required scopes",
                    token_id=token.token_id,
                    required=required_scopes,
                    available=token.scopes,
                )
                return None

        # Check IP allowlist
        if token.allowed_ips and client_ip:
            if not self._is_ip_allowed(client_ip, token.allowed_ips):
                logger.warning(
                    "API key used from disallowed IP",
                    token_id=token.token_id,
                    client_ip=client_ip,
                )
                return None

        # Update last used
        token.last_used_at = datetime.now()

        return token

    # =========================================================================
    # JWT Token Management
    # =========================================================================

    async def create_access_token(
        self,
        user: User,
        tenant_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        audiences: Optional[List[str]] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> TokenGenerationResult:
        """Create a JWT access token."""
        try:
            import jwt
        except ImportError:
            raise RuntimeError("PyJWT is required for JWT tokens. Install with: pip install PyJWT")

        now = datetime.utcnow()
        expires_at = now + self.access_token_expiry

        # Build claims
        claims = {
            "iss": "aion",
            "sub": user.id,
            "aud": audiences or ["aion-api"],
            "iat": now,
            "exp": expires_at,
            "nbf": now,
            "jti": secrets.token_urlsafe(16),
            # Custom claims
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "tenant_id": tenant_id or user.tenant_id,
            "scopes": scopes or [],
        }

        if additional_claims:
            claims.update(additional_claims)

        # Sign token
        token_value = jwt.encode(claims, self.jwt_secret, algorithm=self.jwt_algorithm)

        # Create record for tracking
        token = AuthToken(
            token_type=TokenType.ACCESS,
            token_hash=self._hash_token(token_value),
            token_prefix=token_value[:20],
            user_id=user.id,
            tenant_id=tenant_id or user.tenant_id,
            scopes=scopes or [],
            audiences=audiences or ["aion-api"],
            jti=claims["jti"],
            expires_at=expires_at,
        )

        self._store_token(token)

        return TokenGenerationResult(raw_token=token_value, token_record=token)

    async def create_refresh_token(
        self,
        user: User,
        tenant_id: Optional[str] = None,
        family_id: Optional[str] = None,
    ) -> TokenGenerationResult:
        """
        Create a refresh token.

        Uses token family tracking to detect refresh token reuse attacks.
        """
        try:
            import jwt
        except ImportError:
            raise RuntimeError("PyJWT is required for JWT tokens")

        now = datetime.utcnow()
        expires_at = now + self.refresh_token_expiry
        jti = secrets.token_urlsafe(16)
        family = family_id or secrets.token_urlsafe(16)

        claims = {
            "iss": "aion",
            "sub": user.id,
            "iat": now,
            "exp": expires_at,
            "jti": jti,
            "type": "refresh",
            "family": family,
        }

        token_value = jwt.encode(claims, self.jwt_secret, algorithm=self.jwt_algorithm)

        token = AuthToken(
            token_type=TokenType.REFRESH,
            token_hash=self._hash_token(token_value),
            token_prefix=token_value[:20],
            user_id=user.id,
            tenant_id=tenant_id or user.tenant_id,
            jti=jti,
            expires_at=expires_at,
        )

        self._store_token(token)

        # Track in family
        if family not in self._refresh_families:
            self._refresh_families[family] = []
        self._refresh_families[family].append(token.token_id)

        return TokenGenerationResult(raw_token=token_value, token_record=token)

    async def validate_jwt(
        self,
        token_value: str,
        expected_type: str = "access",
        required_scopes: Optional[List[str]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate a JWT token.

        Returns (claims, error_message).
        """
        try:
            import jwt
        except ImportError:
            return None, "JWT support not available"

        # Check revocation
        token_hash = self._hash_token(token_value)
        if token_hash in self._revoked_hashes:
            return None, "Token has been revoked"

        try:
            claims = jwt.decode(
                token_value,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                audience=["aion-api"],
                options={"require": ["exp", "iat", "sub", "jti"]},
            )

            # Check token type
            if expected_type == "refresh" and claims.get("type") != "refresh":
                return None, "Invalid token type"

            # Check scopes for access tokens
            if required_scopes and expected_type == "access":
                token_scopes = claims.get("scopes", [])
                if not self._has_required_scopes(token_scopes, required_scopes):
                    return None, "Insufficient scopes"

            return claims, None

        except jwt.ExpiredSignatureError:
            return None, "Token has expired"
        except jwt.InvalidAudienceError:
            return None, "Invalid audience"
        except jwt.InvalidTokenError as e:
            return None, f"Invalid token: {str(e)}"

    async def refresh_tokens(
        self,
        refresh_token: str,
        user: User,
    ) -> Tuple[Optional[TokenGenerationResult], Optional[TokenGenerationResult], Optional[str]]:
        """
        Refresh access and refresh tokens.

        Implements refresh token rotation with reuse detection.
        Returns (new_access, new_refresh, error).
        """
        claims, error = await self.validate_jwt(refresh_token, expected_type="refresh")
        if error:
            return None, None, error

        # Get family
        family_id = claims.get("family")
        if not family_id:
            return None, None, "Invalid refresh token format"

        # Check for reuse attack
        refresh_hash = self._hash_token(refresh_token)
        token_id = self._token_hash_index.get(refresh_hash)

        if not token_id:
            # Token not in index - might be reuse of old token
            if family_id in self._refresh_families:
                # Revoke entire family (possible attack)
                await self._revoke_token_family(family_id)
                logger.warning(
                    "Refresh token reuse detected - family revoked",
                    family_id=family_id,
                    user_id=user.id,
                )
                return None, None, "Refresh token reuse detected"
            return None, None, "Invalid refresh token"

        # Revoke old refresh token
        await self.revoke_token(token_id)

        # Create new tokens
        access_result = await self.create_access_token(user)
        refresh_result = await self.create_refresh_token(user, family_id=family_id)

        return access_result, refresh_result, None

    # =========================================================================
    # Token Revocation
    # =========================================================================

    async def revoke_token(
        self,
        token_id: str,
        reason: str = "user_requested",
        revoked_by: Optional[str] = None,
    ) -> bool:
        """Revoke a specific token."""
        token = self._tokens.get(token_id)
        if not token:
            return False

        token.status = TokenStatus.REVOKED
        token.revoked = True
        token.revoked_at = datetime.now()
        token.revoked_by = revoked_by
        token.revocation_reason = reason

        # Add to revocation cache
        self._revoked_hashes.add(token.token_hash)

        logger.info(
            "Token revoked",
            token_id=token_id,
            reason=reason,
            revoked_by=revoked_by,
        )

        return True

    async def revoke_all_user_tokens(
        self,
        user_id: str,
        reason: str = "user_requested",
        except_token_id: Optional[str] = None,
    ) -> int:
        """Revoke all tokens for a user."""
        token_ids = self._user_tokens.get(user_id, [])
        count = 0

        for token_id in token_ids:
            if token_id != except_token_id:
                if await self.revoke_token(token_id, reason):
                    count += 1

        logger.info(
            "Revoked all user tokens",
            user_id=user_id,
            count=count,
            reason=reason,
        )

        return count

    async def _revoke_token_family(self, family_id: str) -> int:
        """Revoke all tokens in a refresh token family."""
        token_ids = self._refresh_families.get(family_id, [])
        count = 0

        for token_id in token_ids:
            if await self.revoke_token(token_id, reason="family_revocation"):
                count += 1

        return count

    # =========================================================================
    # Token Queries
    # =========================================================================

    async def get_token(self, token_id: str) -> Optional[AuthToken]:
        """Get a token by ID."""
        return self._tokens.get(token_id)

    async def get_user_tokens(
        self,
        user_id: str,
        token_type: Optional[TokenType] = None,
        include_revoked: bool = False,
    ) -> List[AuthToken]:
        """Get all tokens for a user."""
        token_ids = self._user_tokens.get(user_id, [])
        tokens = []

        for token_id in token_ids:
            token = self._tokens.get(token_id)
            if not token:
                continue
            if token_type and token.token_type != token_type:
                continue
            if not include_revoked and token.revoked:
                continue
            tokens.append(token)

        return tokens

    async def get_active_api_keys(self, user_id: str) -> List[AuthToken]:
        """Get active API keys for a user."""
        return await self.get_user_tokens(
            user_id, token_type=TokenType.API_KEY, include_revoked=False
        )

    # =========================================================================
    # Token Rotation
    # =========================================================================

    async def rotate_api_key(
        self,
        token_id: str,
        user: User,
    ) -> Optional[TokenGenerationResult]:
        """
        Rotate an API key.

        Creates a new key and schedules the old one for revocation
        after a grace period.
        """
        old_token = self._tokens.get(token_id)
        if not old_token or old_token.token_type != TokenType.API_KEY:
            return None

        # Create new key with same settings
        result = await self.create_api_key(
            user_id=old_token.user_id,
            tenant_id=old_token.tenant_id,
            name=old_token.name,
            description=old_token.description,
            scopes=old_token.scopes,
            expires_in_days=None,  # Calculate from original
            allowed_ips=old_token.allowed_ips,
        )

        # Mark old token for rotation
        old_token.status = TokenStatus.PENDING_ROTATION
        old_token.rotation_count += 1
        old_token.last_rotated_at = datetime.now()

        # Schedule revocation after grace period
        asyncio.create_task(
            self._delayed_revocation(token_id, self.rotation_grace_period.total_seconds())
        )

        logger.info(
            "API key rotated",
            old_token_id=token_id,
            new_token_id=result.token_record.token_id,
        )

        return result

    async def _delayed_revocation(self, token_id: str, delay_seconds: float) -> None:
        """Revoke a token after a delay."""
        await asyncio.sleep(delay_seconds)
        await self.revoke_token(token_id, reason="rotation")

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _store_token(self, token: AuthToken) -> None:
        """Store a token in all indexes."""
        self._tokens[token.token_id] = token
        self._token_hash_index[token.token_hash] = token.token_id

        # Index by user
        if token.user_id:
            if token.user_id not in self._user_tokens:
                self._user_tokens[token.user_id] = []
            self._user_tokens[token.user_id].append(token.token_id)

    def _hash_token(self, token: str) -> str:
        """
        Securely hash a token.

        Uses SHA-256 for consistent hashing.
        """
        return hashlib.sha256(token.encode()).hexdigest()

    def _constant_time_compare(self, a: str, b: str) -> bool:
        """Compare strings in constant time to prevent timing attacks."""
        return hmac.compare_digest(a.encode(), b.encode())

    def _has_required_scopes(
        self,
        available: List[str],
        required: List[str],
    ) -> bool:
        """Check if available scopes satisfy requirements."""
        if "*" in available:
            return True
        return all(scope in available for scope in required)

    def _is_ip_allowed(self, client_ip: str, allowed_ips: List[str]) -> bool:
        """Check if client IP is in allowlist."""
        import ipaddress

        try:
            client = ipaddress.ip_address(client_ip)

            for allowed in allowed_ips:
                if "/" in allowed:
                    # CIDR notation
                    if client in ipaddress.ip_network(allowed, strict=False):
                        return True
                else:
                    if client == ipaddress.ip_address(allowed):
                        return True

            return False
        except ValueError:
            return False

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired tokens."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._cleanup_expired_tokens()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token cleanup error: {e}")

    async def _cleanup_expired_tokens(self) -> None:
        """Remove expired tokens from memory."""
        now = datetime.now()
        expired_ids = []

        for token_id, token in self._tokens.items():
            if token.expires_at and now > token.expires_at + timedelta(days=1):
                expired_ids.append(token_id)

        for token_id in expired_ids:
            token = self._tokens.pop(token_id, None)
            if token:
                self._token_hash_index.pop(token.token_hash, None)
                if token.user_id and token.user_id in self._user_tokens:
                    if token_id in self._user_tokens[token.user_id]:
                        self._user_tokens[token.user_id].remove(token_id)

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired tokens")

    def get_stats(self) -> Dict[str, Any]:
        """Get token manager statistics."""
        active_tokens = sum(
            1 for t in self._tokens.values() if t.status == TokenStatus.ACTIVE
        )
        expired_tokens = sum(
            1 for t in self._tokens.values() if t.status == TokenStatus.EXPIRED
        )
        revoked_tokens = sum(
            1 for t in self._tokens.values() if t.status == TokenStatus.REVOKED
        )

        by_type = {}
        for t in self._tokens.values():
            key = t.token_type.value
            by_type[key] = by_type.get(key, 0) + 1

        return {
            "total_tokens": len(self._tokens),
            "active_tokens": active_tokens,
            "expired_tokens": expired_tokens,
            "revoked_tokens": revoked_tokens,
            "by_type": by_type,
            "unique_users": len(self._user_tokens),
            "revocation_cache_size": len(self._revoked_hashes),
        }
