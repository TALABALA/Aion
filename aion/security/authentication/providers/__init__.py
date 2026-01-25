"""
AION Authentication Providers

Modular authentication provider implementations.
"""

from aion.security.authentication.providers.base import (
    AuthProvider,
    AuthProviderResult,
)
from aion.security.authentication.providers.api_key import APIKeyProvider
from aion.security.authentication.providers.jwt_provider import JWTProvider
from aion.security.authentication.providers.password import PasswordProvider
from aion.security.authentication.providers.oauth import OAuth2Provider

__all__ = [
    "AuthProvider",
    "AuthProviderResult",
    "APIKeyProvider",
    "JWTProvider",
    "PasswordProvider",
    "OAuth2Provider",
]
