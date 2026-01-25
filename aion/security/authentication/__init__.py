"""
AION Authentication Module

Comprehensive authentication system with multiple providers,
token management, session handling, and MFA support.
"""

from aion.security.authentication.service import AuthenticationService
from aion.security.authentication.tokens import TokenManager, TokenGenerationResult
from aion.security.authentication.session import (
    SessionManager,
    SessionConfig,
    SessionSecurityEvent,
)
from aion.security.authentication.providers import (
    AuthProvider,
    AuthProviderResult,
    APIKeyProvider,
    JWTProvider,
    PasswordProvider,
    OAuth2Provider,
)

__all__ = [
    # Main service
    "AuthenticationService",
    # Token management
    "TokenManager",
    "TokenGenerationResult",
    # Session management
    "SessionManager",
    "SessionConfig",
    "SessionSecurityEvent",
    # Providers
    "AuthProvider",
    "AuthProviderResult",
    "APIKeyProvider",
    "JWTProvider",
    "PasswordProvider",
    "OAuth2Provider",
]
