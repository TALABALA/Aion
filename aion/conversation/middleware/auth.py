"""
AION Conversation Authentication Middleware

Handles authentication for conversation endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
import hashlib
import secrets

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class User:
    """Authenticated user."""
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    roles: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.metadata is None:
            self.metadata = {}

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "roles": self.roles,
        }


@dataclass
class AuthToken:
    """Authentication token."""
    token: str
    user_id: str
    expires_at: datetime
    scopes: list[str] = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "*" in self.scopes


class TokenStore:
    """
    In-memory token store.

    For production, use Redis or a database.
    """

    def __init__(self, default_ttl_hours: int = 24):
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self._tokens: dict[str, AuthToken] = {}
        self._users: dict[str, User] = {}

    def create_token(
        self,
        user: User,
        ttl: Optional[timedelta] = None,
        scopes: Optional[list[str]] = None,
    ) -> AuthToken:
        """Create a new authentication token."""
        token_str = secrets.token_urlsafe(32)
        expires_at = datetime.now() + (ttl or self.default_ttl)

        token = AuthToken(
            token=token_str,
            user_id=user.id,
            expires_at=expires_at,
            scopes=scopes or ["*"],
        )

        self._tokens[token_str] = token
        self._users[user.id] = user

        return token

    def validate_token(self, token_str: str) -> Optional[AuthToken]:
        """Validate a token and return it if valid."""
        token = self._tokens.get(token_str)

        if not token:
            return None

        if token.is_expired():
            del self._tokens[token_str]
            return None

        return token

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)

    def revoke_token(self, token_str: str) -> bool:
        """Revoke a token."""
        if token_str in self._tokens:
            del self._tokens[token_str]
            return True
        return False

    def revoke_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a user."""
        to_revoke = [
            token_str
            for token_str, token in self._tokens.items()
            if token.user_id == user_id
        ]

        for token_str in to_revoke:
            del self._tokens[token_str]

        return len(to_revoke)

    def cleanup_expired(self) -> int:
        """Remove expired tokens."""
        now = datetime.now()
        expired = [
            token_str
            for token_str, token in self._tokens.items()
            if token.expires_at < now
        ]

        for token_str in expired:
            del self._tokens[token_str]

        return len(expired)


class AuthMiddleware:
    """
    Authentication middleware for FastAPI.

    Supports:
    - Bearer token authentication
    - API key authentication
    - Optional authentication (for public endpoints)
    """

    def __init__(
        self,
        token_store: Optional[TokenStore] = None,
        api_keys: Optional[dict[str, User]] = None,
        require_auth: bool = True,
    ):
        self.token_store = token_store or TokenStore()
        self.api_keys = api_keys or {}
        self.require_auth = require_auth
        self._bearer = HTTPBearer(auto_error=False)

    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = None,
    ) -> Optional[User]:
        """Authenticate the request."""
        if credentials is None:
            credentials = await self._bearer(request)

        if credentials:
            user = await self._authenticate_bearer(credentials.credentials)
            if user:
                return user

        api_key = request.headers.get("X-API-Key")
        if api_key:
            user = self._authenticate_api_key(api_key)
            if user:
                return user

        if self.require_auth:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return None

    async def _authenticate_bearer(self, token: str) -> Optional[User]:
        """Authenticate using bearer token."""
        auth_token = self.token_store.validate_token(token)

        if not auth_token:
            return None

        user = self.token_store.get_user(auth_token.user_id)
        return user

    def _authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key."""
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        return self.api_keys.get(api_key_hash)

    def require_scope(self, scope: str) -> Callable:
        """Dependency that requires a specific scope."""

        async def check_scope(
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self._bearer),
        ) -> User:
            user = await self(request, credentials)

            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")

            if credentials:
                token = self.token_store.validate_token(credentials.credentials)
                if token and not token.has_scope(scope):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Scope '{scope}' required",
                    )

            return user

        return check_scope

    def require_role(self, role: str) -> Callable:
        """Dependency that requires a specific role."""

        async def check_role(
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self._bearer),
        ) -> User:
            user = await self(request, credentials)

            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")

            if not user.has_role(role):
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{role}' required",
                )

            return user

        return check_role


def create_auth_middleware(
    require_auth: bool = False,
    api_keys: Optional[list[tuple[str, str]]] = None,
) -> AuthMiddleware:
    """
    Create an authentication middleware instance.

    Args:
        require_auth: Whether authentication is required
        api_keys: List of (api_key, user_id) tuples

    Returns:
        Configured AuthMiddleware
    """
    token_store = TokenStore()

    api_key_users = {}
    if api_keys:
        for api_key, user_id in api_keys:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            api_key_users[key_hash] = User(id=user_id, roles=["api"])

    return AuthMiddleware(
        token_store=token_store,
        api_keys=api_key_users,
        require_auth=require_auth,
    )


def get_optional_user(auth: AuthMiddleware) -> Callable:
    """Get a dependency that returns the user or None."""

    async def get_user(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    ) -> Optional[User]:
        try:
            return await auth(request, credentials)
        except HTTPException:
            return None

    return get_user
